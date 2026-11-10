import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { existsSync, readFileSync } from 'node:fs';
import { domFixture, deferred } from './dom-fixture.mjs';
import { mostActive, keptOf, readoutText, readoutTip, READOUT_TIPS } from '../src/cellstats.mjs';
import { runButtons } from '../src/run-buttons.mjs';
import { PHASES, scheduleEnd } from '../src/timeline.js';
import * as region from '../src/region.mjs';
import { LassoDraft, mergedMoves } from '../src/lasso.mjs';
import * as priorOptions from '../src/prior-options.mjs';
import * as stain from '../src/stain.mjs';
import { SETTINGS_TEXT } from '../src/stain-settings.mjs';
import * as proceed from '../src/proceed.mjs';
import * as saveState from '../src/zarr-save-state.mjs';
import { polygonCount } from '../src/polygons.mjs';
import { encodeVectors } from './vectors-fixture.mjs';
import { QcClient } from '../src/qc-client.mjs';
import { pointInOutline, flyTarget, areaOutline } from '../src/qc-areas.mjs';
import { QC_TEXT, copy, duration, cellLine, cardLine, changeLabel, knobName, triedLabel, settingValue, oldProsegTooltip } from '../src/qc-copy.mjs';
import { NEEDS_CLASS, EXPLAIN_TEXT } from '../src/explain.mjs';
import { STAIN_PATCH_TEXT, patchSizeNote } from '../src/stain-patches.mjs';
import { QcPanel as RealQcPanel, withEngine, fromView as realFromView } from '../src/qc-panel.js';
import { clusterColorCSS } from '../src/clusters.js';
import { LEGEND_PLACE } from '../src/qc-overlay.mjs';
import { COMPARE_TEXT } from '../src/compare-copy.mjs';
import { appWindow as tabAppWindow, WINDOW_STORE } from '../src/page-key.mjs';
import { SHELL_TEXT, CLOSE_CHECK_MS } from '../src/shell.mjs';
import { SESSION_TEXT } from '../src/session.mjs';
import { NOTE_TEXT } from '../src/sample-notes.mjs';

// The whole page controller against a fake DOM and fake collaborators: a live dense run opens, frames stream in, the
// pointer rests on a cell, a cell is followed, and the page is hidden and shown again.
const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '');
// The real panels batch, evaluated in the page's context so it uses the page's animation frames and timers.
const paintSource = readFileSync(new URL('../src/paint-batch.mjs', import.meta.url), 'utf8').replace(/^export /gm, '');
const statsSource = readFileSync(new URL('../src/cellstats.mjs', import.meta.url), 'utf8').replace(/^export /gm, '');
const moduleSource = name => readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8')
  .replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
/** Evaluates page module `name` in `context` after the modules it imports from this folder (each once, in its own
 *  scope), and puts its exports into the context: the real stain panels use the real range choosers this way. An
 *  imported module's export never replaces what the page's context already has (a fake panel stays the fake). */
function loadModule(context, name, done = new Set(), top = true) {
  if (done.has(name)) return; done.add(name);
  const text = readFileSync(new URL(`../src/${name}`, import.meta.url), 'utf8');
  for (const [, from] of text.matchAll(/^import [^;]* from '\.\/([^']+)';$/gm)) loadModule(context, from, done, false);
  const names = [...text.matchAll(/^export (?:async )?(?:const|let|function\*?|class) ([A-Za-z_$][\w$]*)/gm)].map(m => m[1]);
  const values = vm.runInContext(`(() => {${moduleSource(name)};\nreturn {${names.join(',')}};})()`, context);
  for (const [key, value] of Object.entries(values)) if (top || !(key in context)) context[key] = value;
}
// The harness's own check that a stain control is clickable, run against the page here in the dev tree; the harness never
// ships, so in production (the deploy runs these tests there) the same test checks the control with the pointer alone.
const stainScenario=new URL('../../harness/scenarios/stain.py',import.meta.url);
const stainControlSource=existsSync(stainScenario)?readFileSync(stainScenario,'utf8').match(/STAIN_CONTROL_JS = """([\s\S]*?)"""/)[1]:null;
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const settle = async (n = 5) => { for (let i = 0; i < n; i++) await new Promise(resolve => setImmediate(resolve)); };
/** Statistics of 8 cells for frame `seq` (what the fake statistics store and the fake engine serve). */
const statsOf = seq => ({ seq, area: new Uint32Array(8).fill(3), assigned: new Uint32Array(8).fill(6),
  foreground: new Uint32Array(8).fill(4), voxels: new Uint32Array(8).fill(9), changed: Uint32Array.from({ length: 8 }, (_, i) => (i * 7 + seq) % 5) });
/** Proseg's count of each of the 8 cells in a counted frame `seq` (its counted file: f32 per cell). */
const countedOf = seq => new Float32Array(8).fill(5.4 + seq / 100);
const statsBuffer = seq => {
  const s = statsOf(seq), buffer = new ArrayBuffer(28 * 8), u = new Uint32Array(buffer);
  ['area', 'voxels', 'assigned', 'foreground', 'changed'].forEach((k, i) => u.set(s[k], i * 8));
  return buffer;
};

/** The mixing check's engine side for the fake run: its answer to GET /runs/{id}/qc and the series, and the calls made. */
function qcEngine(over = {}, { failTrial = false } = {}) {
  const calls = [];
  const describe = { enabled: true, state: 'ready', reason: null, sample_state: 'ready', progress: null, eta_s: null, relearn: false,
    model: { id: 'm1', cells: 15000, created: '2026-10-17T12:00:00Z', seconds: 90 }, start: { per100: 10, judged: 300, mixed: 30, median_tx: 386 },
    start_source: 'xenium', batch_per100: 12, latest_seq: null, seqs: [], thresholds: { mixed_at: .3, judge_min: 100 }, ...over };
  return { calls, describe, api: {
    runQc: async () => { calls.push('describe'); return describe; },
    qcSeries: async () => { calls.push('series'); return { model: 'm1', start: describe.start, points: [] }; },
    setRunQc: async (id, body) => { calls.push(['post', body]); return { ...describe, ...(body.enabled !== undefined ? { enabled: body.enabled } : {}) }; },
    qcSummary: async () => { calls.push('summary'); throw Object.assign(new Error('Not checked yet'), { status: 404 }); },
    qcCells: async (id, seq) => { calls.push(['cells', seq]); return Uint8Array.of(10, 80, 255, 30, 0, 90, 76); },
    learnSampleQc: async () => ({ state: 'waiting', reason: 'queue' }),
    // The trial endpoints.
    qcTrial: async (id, body) => { calls.push(['trial', body]); if (failTrial) throw Object.assign(new Error('A trial is already running'), { status: 409 });
      return { id: 1, ...body, asked_seq: 1, asked_iter: 1, start_seq: null, start_iter: null, rows: [], verdict: null, notes: [] }; },
    qcTrialEnd: async (id, outcome, rule) => { calls.push(['end', outcome, rule ?? null]); return { id: 1, outcome }; },
  } };
}
const qcSummary = (seq, extra = {}) => ({ seq, iter: seq, phase: 'sampling', model: 'm1', single: false, judged: 320, mixed: 35,
  raw_per100: 10.9, per100: 10.9, shown: 11, start: { per100: 10 }, median_tx: 340, batch_per100: 12, areas: [], more_spots: 0, ...extra });

/** realStats: the page's own CellStats (cellstats.mjs) against the fake engine; frameGates: seq -> deferred, the frame
 *  download of that seq is in flight until it is resolved; realPanel: the mixing check's own panel (qc-panel.js) in the
 *  page's #qc-panel instead of one that records what it is given; realAnalysis: the page's own Clusters & UMAP
 *  controller (analysis.js, with a UMAP view that draws nothing) instead of one that records nothing; apiOverride: the
 *  fake engine's answers a test changes from the start (e.g. one that never comes); routes: url -> the fake engine's
 *  answer to a fetch of it, or null for the usual one. */
function page({ realStats = false, realStain = false, realPatches = false, realRoundtrip = false, stainFetch,
  frameGates = new Map(), qc = qcEngine(), runOverride = {}, hook = false, realPanel = false, realAnalysis = false, apiOverride = {},
  routes = null, stageError = null, key = 'k', keyReason = 'signed-in', search = '', tabStorage = null } = {}) {
  const dom = domFixture(), { document, Element } = dom;
  document.body.innerHTML = html;
  class FakeImage { decode() { return calls.imageGate ? calls.imageGate.promise : new Promise(resolve => setTimeout(resolve, 40)); } }
  for (const node of document.body.all()) if (node.attributes?.has('open')) node.open = true;
  Object.assign(Element.prototype, {
    after(node) { const siblings = this.parentElement.childNodes; siblings.splice(siblings.indexOf(this) + 1, 0, node); node.parentElement = this.parentElement; },
    getContext: () => new Proxy({}, { get: (t, k) => k in t ? t[k] : k === 'measureText' ? () => ({ width: 40 }) : () => {} }),
    offsetWidth: 140, offsetHeight: 44, clientWidth: 200, clientHeight: 56,
  });
  Object.defineProperty(Element.prototype, 'firstElementChild', { get() { return this.children[0]; }, configurable: true });
  if (realPanel) Element.prototype.ownerDocument = document;   // the panel builds its nodes with its host's document
  document.querySelectorAll = () => [];
  document.hidden = false;
  const calls = { knobs: 0, setFrame: [], history: 0, selection: [], stats: [], panel: [], chip: [], mixing: [], levels: [], lane: [], legend: [], autoPrior: [],
    tries: [], undos: [], controls: [], optionsSet: null, cells: [], squares: [], startAsks: [], outlines: [] };
  const frame = (seq) => ({ seq, iter: seq, phase: 'burnin', file: '', nx: 2, ny: 2, nz: 1, x0: 0, y0: 0, voxel: 2, temperature: 1,
    loglik: -10 - seq, assigned: 5, foreground: 6, transcripts: 10, cells: 3, visible: 3, changed: seq, t: seq * 0.5, ms: { morphology: 1, repositioning: 1, parameters: 1 }, live: { temperature: 1 } });
  const run = { id: 'r1', sample: 's', sample_name: 'Sample', sample_kind: 'zarr', box: [0, 0, 4, 4], state: 'running', schedule: 'open',
    created: 0, frames: 2, info: { layers: 1, schedule: { burnin: 1, samples: 1, hillclimb: 1 }, plan: [], z: {} }, ack: { state: 'running' }, options: {}, ...runOverride };
  let socket, analysisSelect, telemetry, snapshot;
  const observers = [];   // ResizeObservers: the tests report sizes as the browser would
  const context = vm.createContext({
    ...region, LassoDraft, mergedMoves, ...priorOptions, ...stain, ...saveState, ...proceed, mostActive, keptOf, readoutText, readoutTip, runButtons, PHASES, scheduleEnd, document, Intl, Math, JSON, Promise, Map, Set, Uint32Array, Float32Array, ArrayBuffer, Date, Number, String, Object, Array, Error,
    URLSearchParams, AbortController, performance, setTimeout, clearTimeout, setInterval(fn,ms) { (calls.intervals ??= []).push({fn,ms}); }, confirm: () => true, console,
    harness: realStain || realPatches || hook ? { provide(fn) { snapshot = fn; }, qcShown() {}, count() {} } : null,
    polygonCount,
    Telemetry: class { constructor(options) { telemetry = options; } start() { return this; } },
    ResizeObserver: class { constructor(fn) { this.fn = fn; this.nodes = new Set(); observers.push(this); } observe(node) { this.nodes.add(node); } unobserve(node) { this.nodes.delete(node); } disconnect() { this.nodes.clear(); } },
    requestAnimationFrame: fn => setImmediate(() => fn(performance.now())), cancelAnimationFrame: clearImmediate,
    // The page's own window: its listeners are kept (the window's ✕: beforeunload and pagehide, shell.mjs watchClosing).
    window: { devicePixelRatio: 1, listeners: new Map(), addEventListener(type, fn) { this.listeners.set(type, [...(this.listeners.get(type) || []), fn]); } },
    history: { replaceState() {} }, location: { search, pathname: '/' },
    localStorage: { getItem: () => null, setItem() {} },
    // The sample's overview image takes a while to decode (the picker is usable meanwhile).
    Image: FakeImage,
    Option: class { constructor(text, value) { const option = document.createElement('option'); option.textContent = text; option.value = value; return option; } },
    fetch: async (url) => {
      const routed = routes?.(url); if (routed) return routed;
      const stats = /\/cellstats\/(\d+)$/.exec(url), counted = /\/counted\/(\d+)$/.exec(url);
      if (stats) { calls.stats.push(Number(stats[1])); return { ok: true, arrayBuffer: async () => statsBuffer(Number(stats[1])) }; }
      if (counted) { (calls.counted ??= []).push(Number(counted[1])); return { ok: true, arrayBuffer: async () => countedOf(Number(counted[1])).buffer }; }
      calls.history++; return { ok: true, json: async () => ({ iter: [0, 1, 2], area_um2: [4, 5, 6], transcripts: [1, 2, 3] }) };
    },
    api: {
      health: async () => ({ proseg: true }), samples: async () => [], runs: async () => [{ ...run }], options: async () => [], optionsGuide: async () => [],
      run: async () => ({ ...run }), frames: async () => ({ from: 0, frames: run.kind==='stain' ? [] : [frame(0), frame(1)] }), cells: async () => ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
      control: async (id, action) => { calls.controls.push(action); if (calls.failPause && action === 'pause') throw Object.assign(new Error('The run is finishing'), { status: 409 });
        return { ...run, state: action === 'pause' ? 'paused' : 'running' }; },
      leave: async (id,answer) => { (calls.leaves??=[]).push([id,answer]); if(answer==='stay')return {...run}; return {...run,leaving:{answer,stage:'writing',message:'Writing…'}}; },
      createRun: async (...args) => { calls.created = args; return { ...run, id: 'r9' }; },
      sample: async id => ({ id, name: 'Sample', transcripts: 1000, extent: [0, 0, 2000, 2000], shapes: [], limits: { detail_um: 1200 },
        overview: { x0: 0, y0: 0, width: 10, height: 10, um_per_px: 200 },
        prior_options: { cells: [{ kind: 'column', column: 'cell_id', label: 'cell_id' }, { kind: 'column', column: 'other', label: 'other' }], nuclei: [] },
        suggested_prior: { cells: { kind: 'column', column: 'cell_id' } } }),
      shapes: async () => ({polygons:[],count:0}),
      sampleQc: async () => ({ state: 'ready' }), overviewUrl: id => `/overview/${id}`,
      // An image the engine serves, fetched (api.js image): decoded, as main.js draws it.
      image: async url => { (calls.images ??= []).push(url); const img = new FakeImage(); await img.decode(); return img; },
      frame: async (id, seq) => { if (frameGates.has(seq)) await frameGates.get(seq).promise; return new Uint32Array(12).buffer; }, imageUrl: () => '',
      ...qc.api, ...apiOverride,
    },
    // The mixing check: the real client (qc-client.mjs) and copy helpers; the panel, the chip, the overlay and the
    // legend record what they were given. The overlay takes the per-bin layer's level, never an area: no area is drawn
    // since 2026-10-22 (it has no setAreas, hit, hover, select or target: a call to one would throw).
    QcClient, QC_TEXT, copy, duration, cellLine, cardLine, changeLabel, knobName, triedLabel, settingValue, fetchBins: async () => null, pointInOutline, flyTarget,
    areaOutline, clusterColorCSS,
    LEGEND_PLACE,
    fromView: (view, page) => view ? { state: view.status, enabled: view.enabled, summary: view.summary, start: view.start, atStart: view.atStart,
      suggestion: view.suggestion, trial: view.trial, trialEnded: view.trialEnded, ...page } : { state: 'loading' },
    withEngine,
    QcPanel: realPanel ? RealQcPanel : class { constructor(host, o) { this.options = o; } setState(s) { calls.panel.push(s); this.state = s; } open() { calls.panel.push('open'); }
      setCovered(on) { calls.panelCovered = on; }   // Compare over the page: its test is not asked for meanwhile
      // The selected cell's row (qc-panel.js setCell): what it is given, and its one button bound as the panel binds it.
      setCell(cell) { calls.cells.push(cell); if (cell) this.options.bindDeeper?.(this.deeperButton ??= document.createElement('button'), cell.label); } },
    QcChip: class { constructor(host, o) { this.onClick = o?.onClick; } setState(s) { calls.chip.push(s); this.state = s; } get visible() { return ['live', 'paused'].includes(this.state?.state); } },
    QcOverlay: class { constructor() { this.anchor = null; } draw() {} clear() {}
      overview() { return false; } binStops() { return [[.1, '#000'], [.15, '#111'], [.2, '#222']]; }
      wantsBins() { return false; } setBins() { return false; }
      setLevel(o) {
        calls.levels.push([o, !!this.calm]);
        const anchor = Number.isFinite(o?.anchorPer100) ? o.anchorPer100 : Number.isFinite(o?.tissuePer100) ? o.tissuePer100 : null;
        const changed = anchor !== this.anchor; this.anchor = anchor; return changed;
      }
      // The test's square, as set ({box} | null; QcOverlay.testSquare: the box or null).
      setTestSquare(sq) { calls.squares.push(sq); this.square = sq?.box ?? null; return true; } get testSquare() { return this.square ?? null; } },
    // The legend: what it shows; main.js places its root out of the test corner's way (its style: left, bottom, visibility).
    MixingLegend: class { constructor() { this.root = { hidden: false, style: { left: '12px', bottom: '12px', visibility: '' } }; } update(o) { calls.legend.push(o); } },
    // Go deeper (qc-deeper.js, qc-overlay.mjs DeeperArrows): the window and its arrows record what main.js hands them.
    QcDeeper: class {
      constructor(host, o) { Object.assign(this, { host, options: o, root: { hidden: true }, selects: [], events: [], changes: 0, resets: [], toggles: [], hovers: [], bound: [] }); calls.deeper = this; }
      reset(run) { this.resets.push(run?.id ?? null); } select(label) { this.selects.push(label); } changed() { this.changes++; } event(m) { this.events.push(m); }
      refresh() { this.refreshed = (this.refreshed ?? 0) + 1; } layout() {} hover(label) { this.hovers.push(label); } tooltipLine(label) { return label === this.sourceLabel ? 'c2 · makes GXPN: about 46 % of c6\'s RNA seems to come from here (estimate)' : null; }
      setStageNote() {} snapshot() { return { open: false, cell: null }; } texts() { return ['Go deeper words']; }
      setCovered(on, catchUp = true) { this.covered = on; (this.covers ??= []).push([on, catchUp]); }   // Compare over the page
      bindButton(button, label) { button.textContent = 'Go deeper ›'; button.dataset.action = 'go-deeper'; this.bound.push(label); button.onclick = () => this.toggle(label); }
      toggle(label) { this.toggles.push(label); } open(label, o) { (this.opens ??= []).push(label); return true; }
      // Its proseg line follows the frame's statistics; its suspects of a cell go with a test of a setting.
      statsChanged() { this.statsChanges = (this.statsChanges ?? 0) + 1; } suspects(label) { return this.kept?.cell === label ? this.kept.list : []; } },
    DeeperArrows: class { constructor(stage, o) { this.options = o; this.missing = false; calls.arrows = this; this.sets = []; }
      set(focus, o) { this.sets.push([focus, o]); return false; } clear() { return false; } sync() {} draw() {} hit() { return 0; } texts() { return ['46 %']; } },
    // The test corner (qc-test-corner.mjs): the models the panel hands it, its layouts and motion refreshes; its options are
    // main.js's wiring (fetchView, onAction, onBox, the motion settings, the stage's size).
    TestCorner: class { constructor(host, o) { Object.assign(this, { host, options: o, root: { hidden: false }, models: [], layouts: 0, refreshes: 0, recolors: 0, place: null }); calls.corner = this; }
      show(model) { this.models.push(model); } layout() { this.layouts++; } refresh() { this.refreshes++; } texts() { return ['Suspects inside: 2 of 3']; }
      setCovered(on) { this.covered = on; }   // Compare over the page
      // Its box for Go deeper's camera move (null while closed), and its cells drawn again in new colours.
      box() { return this.place; } recolor() { this.recolors++; }
      snapshot() { return { open: this.models.at(-1)?.open === true, phase: this.models.at(-1)?.phase ?? 'closed' }; } },
    // Compare (compare.mjs): how main.js opens the view and how it closes (Back calls onClose; a run or the
    // picker opening closes it without); its words are the real deck's.
    CompareView: class { constructor(host, o) { Object.assign(this, { host, options: o, isOpen: false, opens: [], refreshes: 0, root: document.createElement('dialog') }); calls.compare = this; }
      open(args) { this.opens.push(args); this.isOpen = true; return true; } close(tell = true) { if (!this.isOpen) return; this.isOpen = false; if (tell) this.options.onClose?.(); }
      refresh() { this.refreshes++; } texts() { return this.isOpen ? ['13 in 100 cells look mixed'] : []; }
      snapshot() { return { open: this.isOpen, state: this.isOpen ? 'working' : 'closed', columns: [], area: null }; } },
    COMPARE_TEXT,
    // The live socket: the tests hand main.js's handlers messages; its goodbye (pagehide, Close), its return (pageshow)
    // and its stop are recorded; it is open (connected) unless a test says otherwise.
    LiveSocket: class { constructor(handlers) { socket = handlers; calls.socket = []; } watch() {} goodbye() { calls.socket.push('goodbye'); }
      back() { calls.socket.push('back'); } stop() { calls.socket.push('stop'); } connected() { return calls.socketOpen ?? true; } },
    // The tab's sign-in (page-key.mjs): signed in unless a test says otherwise; whether it is the app's own window.
    pageKey: { ready: Promise.resolve(key), reason: () => keyReason }, SIGNED_OUT: 'sopa-signed-out',
    appWindow: () => tabAppWindow({ location: { search }, sessionStorage: tabStorage }),
    MARK: 'sopa-companion:api-session', sessionStorage: tabStorage,
    Options: class { constructor() { this.schema = []; } load(schema) { this.schema = schema; } getValues() { return {}; } problems() { return []; }
      setAll(values, o) { calls.optionsSet = values; calls.explicit = o?.explicit; calls.optionsSaved = o?.save !== false; return Object.keys(values); }
      save() { calls.optionsSaved = true; } restore() { calls.restored = (calls.restored ?? 0) + 1; } },
    LiveKnobs: class { render(r) { calls.knobs++; this.key = `${r?.id}:`; }
      // The Live knobs session (options.js): the mixing check's Try it and Undo go through it.
      async trySetting(flag, value, from) { calls.tries.push([flag, value, from]); await new Promise(resolve => setImmediate(resolve)); if (this.refuse) throw new Error(this.refuse); return { value: 0.04, automatic: false }; }
      async undoSetting(flag, value) { calls.undos.push([flag, value]); this.beforeUndo?.(); return { value }; } },
    // The knob guides (options.js): what main.js hands them, and how often it asks them to write their lines again.
    KnobGuide: class { constructor(host, toggle, o) { this.options = o; this.updates = 0; } load() {} update() { this.updates++; } },
    Stage: class {
      constructor(canvas) { Object.assign(this, { canvas, camera: { x: 0, y: 0, zoom: 1 }, orbit: { active: false, enabled: false, switch() {} }, options: { mode: 0, hover: 0 },
        cssWidth: 1000, cssHeight: 800, ratio: 1, scale: 1, listeners: [] }); canvas.width = 1000; canvas.height = 800; }
      set(o) { Object.assign(this.options, o); } setFrame(f) { this.frame = f; calls.setFrame.push(f?.iter); } setImage() {} fit(rect,animate){(calls.fits??=[]).push([rect,animate]);} resize() {}
      setCovered(on) { this.covered = on; }   // Compare over the page: nothing drawn until it goes
      setMixing(bytes, o) { calls.mixing.push(bytes); calls.mixingOptions = o; calls.mixingCalm = !!this.calmMotion; }
      onChange(fn) { this.listeners.push(fn); } toWorld(x, y) { return [x, y]; } toScreen(x, y) { return [x, y]; } pickLabel() { return 7; }
      setAnalysis(result) { this.analysis = result; }   // the clusters on the tissue (the real Clusters & UMAP controller)
      // Go deeper's starting outline, kept as the stage keeps it ({points, closed, count} | null).
      setStartOutline(o) { calls.outlines.push(o); this.startOutline = o ? { points: o.points, closed: o.closed !== false, count: o.points.length / 2 } : null; return true; }
    },
    TranscriptLoader: class { reset() {} load() {} },
    // The prior's layers; a cell's starting outline by its name and label, once the prior holds it.
    PriorLayers: class { constructor(){this.settings={xenium:{},nuclei:{},prior:{}};this.starts=new Map();
        // The sample's segmentation layers (seg-layers.mjs): the samples main.js hands the store, in order.
        this.segments={sample:null,samples:[],saved:[],setSample(id){this.sample=id;this.samples.push(id);},sampleSaved(id){this.saved.push(id);}};} reset() {} load() {}
      startOutline(name, label) { calls.startAsks.push([name, label]); return this.starts.get(name) ?? null; }
      segmentationsShown() { return { sample: this.segments.sample, note: null, loaded: [] }; } },
    StainView: realStain ? stain.StainView : class { constructor(){calls.stain=[];} reset(){} refresh(){} update(run){calls.stain.push(run.progress?.done);} },
    SETTINGS_TEXT, STAIN_PATCH_TEXT, patchSizeNote,
    TissueView: class { polygons=[]; update(){} },
    StainExclusions: class { constructor(stage,api,options){calls.exclusions=options;} reset(){} update(){} draw(){} setVectors(){} },
    CellposeSettings: class { refresh(){} },
    StainPatches: class { settings={}; settingsReason={}; reset(){} update(){} move(){} load(){} snapshot(){return {active:false};}
      // The picker's suggested boxes (stain-patches.mjs showPreview): kept as shown.
      showPreview(result){calls.previews=(calls.previews??0)+1;this.boxes=result.patches.map(p=>[...p.box]);} previewBoxes(){return this.boxes??null;}
      clearPreview(){calls.previewClears=(calls.previewClears??0)+1;this.boxes=null;} },
    LayerPanel: class { constructor() { this.saved = {}; } render() {} setChannels() {} save() {} autoPrior(on) { calls.autoPrior.push(on); } setMixingAvailable(on, why = null) { this.mixingAvailable = on; this.mixingWhy = why; }
      mountSegmentations(host) { (calls.segHosts ??= []).push(host?.id ?? null); } },
    TiledLoader: class { constructor(){this.requests={jobs:new Map()};} reset() {} imageView(){} }, XeniumMasks: class { reset() {} pick() { return []; } render() {} },
    // The statistics store's contract (cellstats.mjs): prefetch fills the cache only; load makes a frame's statistics
    // current (at once when cached) and notifies.
    ...(realStats ? {} : { CellStats: class {
      constructor(changed) { this.changed = changed; this.current = null; this.cache = new Map(); }
      reset() { this.current = null; this.cache = new Map(); }
      prefetch(seq) { return Promise.resolve().then(() => { this.cache.set(seq, statsOf(seq)); return this.cache.get(seq); }); }
      load(seq) {
        const cached = this.cache.get(seq);
        if (cached) { if (this.current !== cached) { this.current = cached; this.changed(); } return Promise.resolve(cached); }
        this.current = null; return this.prefetch(seq).then(s => { this.current = s; this.changed(); return s; });
      }
    } }),
    Analysis: class { constructor(stage, panel, ctx, select) { analysisSelect = select; } reset() {} refresh() {} frame() {} event() {} selection(l) { calls.selection.push(l); } tissueHover() {} loadStats() { return Promise.resolve(null); }
      setCovered(on, catchUp = true) { this.covered = on; (this.covers ??= []).push([on, catchUp]); } },   // Compare over the page: no result applied meanwhile
    Timeline: class { constructor() { this.current = -1; this.urgent = []; } setFrames(f) { this.frames = f; } setPlan() {} setCurrent(s, urgent) { this.current = s; this.urgent.push(!!urgent); } refresh() {} get total() { return 0; }
      setQc(lane) { calls.lane.push(lane); } setCovered(on) { this.covered = on; } },
    SampleBrowser: class { show() { calls.browserOpened = (calls.browserOpened || 0) + 1; } }, ZarrSave: class { constructor(onRun, notice, onSaved, whenSettled, onCancel) { this.pending = new Map(); this.onSaved = onSaved; this.onCancel = onCancel; calls.zarrSave = this; } update() {} },
  });
  // Buttons that say what is missing (explain.mjs), evaluated here so their notes use this page's document.
  Object.assign(context, vm.runInContext(`(() => {${moduleSource('explain.mjs')};
    return {problem,explain,guard,busy,clearExplanation,EXPLAIN_TEXT,NEEDS_CLASS};})()`, context));
  // The app shell (shell.mjs): the first screen's tools and the leave prompt; the older runs' prompt (older-runs.mjs)
  // and a notebook's session (session.mjs), built in this page's document.
  for (const name of ['shell.mjs', 'older-runs.mjs', 'session.mjs']) loadModule(context, name, new Set(['zarr-save-state.mjs']));
  // What the page says about a sample (sample-notes.mjs: the picker's notes and the tools it offers), in this document.
  Object.assign(context, vm.runInContext(`(() => {${moduleSource('sample-notes.mjs')};
    return {NOTE_TEXT,openFailure,sampleTools,sampleNotes,renderNotes};})()`, context));
  if(realPatches) {
    document.createElementNS=(_,tag)=>document.createElement(tag);
    // The panel's export dialog (patch-folder.mjs), on this page's explain.mjs and sample-notes.mjs (above).
    loadModule(context, 'patch-folder.mjs', new Set(['explain.mjs', 'sample-notes.mjs']));
    Object.assign(context,vm.runInContext(`(() => {${moduleSource('stain-patches.mjs')};
      return {StainPatches,canReviewStain};})()`,context));
    context.window.fetch=stainFetch;
    context.window.dispatchEvent=()=>{};
    context.CustomEvent=class {constructor(type,options){Object.assign(this,{type,...options});}};
    const realApi=vm.runInContext(`(() => {${moduleSource('api.js')};return api;})()`,context);
    context.fetch=(...args)=>context.window.fetch(...args);
    for(const key of Object.keys(realApi).filter(k=>k.startsWith('stain')))context.api[key]=realApi[key];
  }
  if(realRoundtrip) {
    context.TextDecoder=TextDecoder;   // the cell values' metadata (stain-cells.mjs decodeCellValues), as a browser has it
    Object.assign(context,vm.runInContext(`(() => {${moduleSource('stain-settings.mjs')};
      return {TissueView,CellposeSettings,fillCellposeSettings,settingsReport};})()`,context));
    // The real removed-cells panel and what it imports (stain-patches.mjs, range-chooser.mjs, stain-cells.mjs...).
    const known = new Set(['explain.mjs', 'region.mjs', 'lasso.mjs', 'stain-settings.mjs']);
    loadModule(context, 'stain-exclusions.mjs', known);
    context.api.cellposeSettings=async()=>({found:false});
    context.api.stainExclusions=async()=>({version:0,areas:[],removed:0});
    context.api.tissue=async()=>({source:'region_of_interest',polygons:[[[0,0],[100,0],[100,100]]],area_um2:5000});
  }
  if (realAnalysis) {
    // The page's own Clusters & UMAP controller in this page's document: it splits the stage's box to hold its pane, so
    // a node moved into another is taken out of the one it was in, as in a browser.
    const query = document.querySelector, append = Element.prototype.append;
    document.querySelector = selector => selector === '.stage-wrap' ? document.body.all().find(node => node.className === 'stage-wrap') ?? null : query(selector);
    Element.prototype.before = function (node) { const siblings = this.parentElement.childNodes; siblings.splice(siblings.indexOf(this), 0, node); node.parentElement = this.parentElement; };
    Element.prototype.append = function (...children) {
      for (const child of children) if (child && typeof child === 'object' && child.parentElement?.childNodes.includes(child)) child.remove();
      return append.apply(this, children);
    };
    Object.assign(context, { pageLog: { add() {}, provide() {} },
      UmapView: class { constructor(canvas, handlers) { this.handlers = handlers; this.hover = null; } setData() {} resize() {} dispose() {} hideClusters() {}
        highlightCells() {} highlightCluster() {} select() {} setCovered() {} } });
    loadModule(context, 'analysis.js', new Set(['umap.js', 'harness-hook.mjs', 'log-export.mjs', 'explain.mjs', 'cellstats.mjs']));
  }
  if (stageError) context.Stage = class { constructor() { throw stageError; } };   // a browser that cannot draw
  const errors = [];
  if (stageError) {
    let thrown = null;
    try { vm.runInContext(paintSource + source, context); } catch (error) { thrown = error; }
    return { dom, $: dom.$, context, calls, thrown };
  }
  vm.runInContext((realStats ? statsSource : '') + paintSource + source + '\nthis.__main = { state, stage, timeline, analysis, cellStats, shown, paint, qcClient, layerPanel, priors, stainView, stainPatches, stainExclusions, tissueView, pickerSettings };', context);
  return { dom, $: dom.$, context, calls, frame, observers, socket: () => socket, select: id => analysisSelect(id), errors, main: () => context.__main, telemetry: () => telemetry, snapshot: () => snapshot(), qc };
}


async function discardRun(p) {
  await settle();
  assert.equal(p.$('leave-popup').open,true);
  await p.$('leave-discard').click();
  p.socket().message({ev:'gone',run:p.main().state.run.id,reason:'left'});
  await settle();
}
test('a live run streams frames, keeps its panels current and rewrites only what changed', async () => {
  const p = page();
  await settle(20);
  const { state, timeline } = p.main();
  assert.equal(state.mode, 'run'); assert.equal(state.current, 1);
  assert.equal(p.$('hud-iter').textContent, '1');
  assert.match(p.$('status').textContent, /^Iteration 1 · /);   // iterations, as the HUD names them: never snapshot numbers
  assert.equal(p.$('notice').hidden, true, p.$('notice').textContent);
  const knobRenders = p.calls.knobs, facts = p.$('facts').children;
  assert.ok(facts.length > 10);
  assert.deepEqual(p.$('active-cells').children.map(b => b.tagName), ['STRONG', ...Array(6).fill('BUTTON')], 'the six cells that changed');

  // A resting pointer: the tooltip shows the cell under it and follows each new frame's numbers.
  p.$('stage').emit('pointermove', { offsetX: 980, offsetY: 30, pointerId: 1, button: 0 });
  const tip = p.$('tooltip');
  assert.equal(tip.hidden, false); assert.equal(tip.children[0].textContent, 'c6');
  assert.equal(tip.style.left, `${1000 - 140 - 8}px`, 'clamped inside the cached stage size');

  for (let seq = 2; seq < 6; seq++) {
    p.socket().message({ ev: 'frames', run: 'r1', from: seq, frames: [p.frame(seq)] });
    await settle(20);
  }
  assert.equal(state.current, 5); assert.equal(p.$('hud-iter').textContent, '5');
  assert.deepEqual(p.calls.setFrame.slice(-4), [2, 3, 4, 5]);
  assert.ok(timeline.urgent.slice(-4).every(u => !u), 'a live playhead is not an urgent timeline redraw');
  assert.equal(p.calls.knobs, knobRenders, 'live knobs are not re-rendered for frames that change nothing they show');
  assert.equal(p.$('facts').children[0], facts[0], 'facts are updated in place');
  assert.match(p.$('facts').textContent, /Iteration5/);

  // Follow a cell: the card shows its numbers and whole-run history; frames update it without refetching every time.
  p.select(6);
  await settle(20);
  assert.equal(state.selected, 7); assert.ok(p.calls.selection.includes(7));
  const card = p.$('cell-card');
  assert.equal(card.hidden, false); assert.match(card.textContent, /c6/); assert.match(card.textContent, /Area \(µm²\)/);
  const fetched = p.calls.history, header = card.children[0];
  p.socket().message({ ev: 'frames', run: 'r1', from: 6, frames: [p.frame(6)] });
  await settle(20);
  assert.equal(p.calls.history, fetched, 'history is refreshed at most every HISTORY_MS');
  assert.equal(card.children[0], header, 'the card is updated in place');

  assert.equal(p.$('runs-button'), null);

  // A hidden page follows nothing; showing it again catches up with the newest frame.
  p.dom.document.hidden = true;
  p.socket().message({ ev: 'frames', run: 'r1', from: 7, frames: [p.frame(7), p.frame(8)] });
  await settle(20);
  assert.equal(state.current, 6);
  p.dom.document.hidden = false;
  for (const fn of p.dom.document.listeners.get('visibilitychange') || []) fn();
  await settle(20);
  assert.equal(state.current, 8);
  assert.equal(p.$('notice').hidden, true, p.$('notice').textContent);
  vm.runInContext('state.selected = 0; updateCellCard();', p.context);   // releases the pending history refresh
  assert.equal(p.$('cell-card').hidden, true);
});

test('statistics arriving under a resting pointer re-measure the tooltip, so it stays inside the stage', async () => {
  const p = page();
  await settle(20);
  const { cellStats } = p.main(), tip = p.$('tooltip');
  // Its width follows its text: 7 px a character, CSS max-width 360 px plus 22 px of padding and border.
  Object.defineProperty(tip, 'offsetWidth', { get: () => 22 + Math.min(360, 7 * Math.max(...tip.children.map(c => c.textContent.length))) });
  const load = cellStats.load.bind(cellStats), gate = deferred();
  cellStats.load = seq => { cellStats.current = null; return gate.promise.then(() => load(seq)); };
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  p.$('stage').emit('pointermove', { offsetX: 960, offsetY: 30, pointerId: 1, button: 0 });   // near the right edge
  assert.equal(tip.children[1].textContent, 'Statistics unavailable');
  const before = parseInt(tip.style.left);
  gate.resolve(); await settle(20);                                       // the frame's statistics: a long readout
  assert.match(tip.children[1].textContent, /µm² · /);
  const left = parseInt(tip.style.left);
  assert.ok(left < before && left + tip.offsetWidth <= 1000 - 8, `left ${left}, width ${tip.offsetWidth}`);
  p.$('stage').emit('pointermove', { offsetX: 950, offsetY: 30, pointerId: 1, button: 0 });   // same cell, same text
  assert.equal(tip.style.left, `${left}px`);
});

test('the frame on screen keeps its readout while the next snapshot downloads; one statistics request per frame', async () => {
  const frameGates = new Map(), p = page({ realStats: true, frameGates });
  await settle(20);
  const { state, cellStats } = p.main(), tip = p.$('tooltip');
  const hover = () => { p.$('stage').emit('pointermove', { offsetX: 100, offsetY: 30, pointerId: 1, button: 0 }); return tip.children[1]?.textContent; };
  assert.equal(state.current, 1); assert.equal(cellStats.current?.seq, 1);
  const idle = hover();
  // Not a counted frame: its non-background count, in the words of proseg's count: as saved (one draw, H8).
  assert.equal(idle, "12 µm² · 4 in proseg's count (as saved) · 2 set aside as background · 9 voxels · Unanalysed");
  const active = p.$('active-cells').textContent;
  frameGates.set(2, deferred());                     // live: snapshot 2 is announced, its download is in flight
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  assert.equal(state.current, 1, 'frame 1 is still on screen');
  assert.equal(hover(), idle, 'with its own numbers (not "Statistics unavailable")');
  assert.equal(cellStats.current.seq, 1); assert.equal(p.$('active-cells').textContent, active);
  frameGates.get(2).resolve(); await settle(20);
  assert.equal(state.current, 2); assert.equal(cellStats.current.seq, 2);
  assert.equal(hover(), idle);                       // frame 2's numbers (the same values here)
  assert.notEqual(p.$('active-cells').textContent, active, 'the most active cells changed with the snapshot');
  assert.deepEqual(p.calls.stats, [1, 2], 'one statistics request per frame shown');
  assert.equal(p.calls.counted, undefined, 'no frame known to be counted: no count asked for');
});

test("a counted frame's readouts read proseg's own count of the cell (its counted file), Go deeper's basis too; one learned to be counted later gets it then", async () => {
  // Frame 1's record names its counted file; frame 2's counts line comes after its record (the run's proseg_counts).
  const frameGates = new Map(), p = page({ realStats: true, frameGates });
  await settle(20);
  const { state, cellStats } = p.main(), tip = p.$('tooltip');
  state.frames[1] = { ...state.frames[1], counted_file: 'frames/000001.counted.zst' };
  cellStats.reset('r1'); vm.runInContext('show(1)', p.context); await settle(20);
  const hover = () => { p.$('stage').emit('pointermove', { offsetX: 100, offsetY: 30, pointerId: 1, button: 0 }); return tip.children[1]?.textContent; };
  assert.equal(state.current, 1); assert.deepEqual([...cellStats.current.kept], [...countedOf(1)]);
  assert.equal(hover(), "12 µm² · 5 in proseg's count (averaged) · 1 set aside as background · 9 voxels · Unanalysed", 'whole numbers: 5.41 kept, 0.59 set aside');
  assert.deepEqual({ ...vm.runInContext('deeperCellStats(7)', p.context) }, { assigned: 6, kept: countedOf(1)[6] }, "Go deeper's basis reads the same count");
  assert.deepEqual(p.calls.counted, [1]);
  // The card's readout line says in its tooltip which count it reads: averaged on a counted frame, one draw elsewhere.
  p.select(6); await settle(20);
  const cardLine = () => p.$('cell-card').children[1];
  assert.equal(cardLine().textContent, hover()); assert.equal(cardLine().title, READOUT_TIPS.kept);
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] }); await settle(20);
  assert.equal(state.current, 2); assert.equal(hover(), "12 µm² · 4 in proseg's count (as saved) · 2 set aside as background · 9 voxels · Unanalysed");
  assert.equal(cardLine().textContent, hover()); assert.equal(cardLine().title, READOUT_TIPS.saved);
  // Go deeper's basis beside frame 1's result reads frame 1's statistics (held here), never the frame on screen's; frame
  // 2's have no counted file yet: no number.
  assert.deepEqual({ ...vm.runInContext('deeperCellStats(7, 1)', p.context) }, { assigned: 6, kept: countedOf(1)[6] });
  assert.equal(vm.runInContext('deeperCellStats(7, 2)', p.context), null);
  const changes = p.calls.deeper.statsChanges ?? 0;
  p.socket().message({ ev: 'run', run: { ...state.run, proseg_counts: { reason: null, newest: 2, recent: [1, 2] } } }); await settle(20);
  assert.deepEqual(p.calls.counted, [1, 2]);
  assert.equal(hover(), "12 µm² · 5 in proseg's count (averaged) · 1 set aside as background · 9 voxels · Unanalysed");
  assert.equal(cardLine().title, READOUT_TIPS.kept, 'and its tooltip with it');
  assert.ok(p.calls.deeper.statsChanges > changes, 'Go deeper hears of it');
  p.socket().message({ ev: 'run', run: { ...state.run, proseg_counts: { reason: null, newest: 2, recent: [1, 2] } } }); await settle(20);
  assert.deepEqual(p.calls.counted, [1, 2], 'asked once');
  // Frame 2 has left the run's newest counted frames (64 of them), and its record was pushed without its counted file:
  // still known to be counted, looked at again (its statistics asked anew), it reads proseg's count.
  p.socket().message({ ev: 'run', run: { ...state.run, proseg_counts: { reason: null, newest: 70, recent: [7, 70] } } }); await settle(20);
  cellStats.reset('r1'); vm.runInContext('show(1)', p.context); await settle(20); vm.runInContext('show(2)', p.context); await settle(20);
  assert.equal(state.current, 2); assert.deepEqual(p.calls.counted, [1, 2, 1, 2]);
  assert.equal(hover(), "12 µm² · 5 in proseg's count (averaged) · 1 set aside as background · 9 voxels · Unanalysed");
  // A page away while more than 64 frames were counted (its socket lost): the watch it sends on reconnecting names
  // every counted frame it holds ({ev: "counted"}), the frame on screen's count is asked for then.
  p.socket().message({ ev: 'run', run: { ...state.run, proseg_counts: { reason: null, newest: 900, recent: [899, 900] } } }); await settle(20);
  state.frames[1] = { ...state.frames[1], counted_file: undefined }; vm.runInContext('countedSeqs.delete(1)', p.context);
  cellStats.reset('r1'); vm.runInContext('show(1)', p.context); await settle(20);
  assert.equal(vm.runInContext('frameCounted(1)', p.context), false);
  assert.equal(hover(), "12 µm² · 4 in proseg's count (as saved) · 2 set aside as background · 9 voxels · Unanalysed", 'one draw while not known');
  p.socket().message({ ev: 'counted', run: 'r2-not-this-one', seqs: [1] }); await settle(20);
  assert.equal(vm.runInContext('frameCounted(1)', p.context), false, "another run's list says nothing of this one");
  p.socket().message({ ev: 'counted', run: 'r1', seqs: [0, 1, 'x'] }); await settle(20);
  assert.equal(vm.runInContext('frameCounted(1)', p.context), true);
  assert.equal(p.calls.counted.at(-1), 1, "the frame on screen's count asked for");
  assert.equal(hover(), "12 µm² · 5 in proseg's count (averaged) · 1 set aside as background · 9 voxels · Unanalysed");
  // Another run opened knows none of it.
  assert.equal(vm.runInContext('frameCounted(2)', p.context), true);
  p.context.api.run = async id => ({ ...state.run, id, proseg_counts: { reason: null, newest: 5, recent: [5] } });
  await vm.runInContext("openRun('r2')", p.context); await settle(20);
  assert.deepEqual([2, 5].map(seq => vm.runInContext(`frameCounted(${seq})`, p.context)), [false, true]);
});

test("a frame counted for a pause, the hold, a finish or the end reads the count proseg saves there: its readouts, the card and Go deeper's basis say as saved", async () => {
  // Decision A, H8: the engine scores such a frame on f; the page's readouts read the frame's own draw (cells.zst
  // foreground, Σf) with the saved words, never its counted file (Σe): none is asked for. Frame 1's record says so
  // ({count: "sampled"}); frame 2 learns it from the run's proseg_counts.sampled after its counted file came (a token's
  // count of it, then the hold's).
  const p = page({ realStats: true });
  await settle(20);
  const { state, cellStats } = p.main(), tip = p.$('tooltip');
  state.frames[1] = { ...state.frames[1], counted_file: 'frames/000001.counted.zst', count: 'sampled' };
  cellStats.reset('r1'); vm.runInContext('show(1)', p.context); await settle(20);
  const hover = () => { p.$('stage').emit('pointermove', { offsetX: 100, offsetY: 30, pointerId: 1, button: 0 }); return tip.children[1]?.textContent; };
  assert.equal(vm.runInContext('frameSampled(1)', p.context), true); assert.equal(cellStats.current.kept, null);
  assert.equal(p.calls.counted, undefined, 'no counted file asked for');
  assert.equal(hover(), "12 µm² · 4 in proseg's count (as saved) · 2 set aside as background · 9 voxels · Unanalysed");
  // Go deeper's basis reads the same count: Σf (cells.zst foreground), not rounded.
  assert.deepEqual({ ...vm.runInContext('deeperCellStats(7)', p.context) }, { assigned: 6, kept: 4 });
  p.select(6); await settle(20);
  const cardLine = () => p.$('cell-card').children[1];
  assert.equal(cardLine().textContent, hover()); assert.equal(cardLine().title, READOUT_TIPS.saved);
  // Frame 2: a token counted it (its counted file comes with its statistics), then the hold counted it again.
  p.socket().message({ ev: 'run', run: { ...state.run, proseg_counts: { reason: null, newest: 2, recent: [1, 2] } } }); await settle(20);
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] }); await settle(20);
  assert.equal(state.current, 2); assert.deepEqual(p.calls.counted, [2]);
  assert.equal(hover(), "12 µm² · 5 in proseg's count (averaged) · 1 set aside as background · 9 voxels · Unanalysed");
  assert.deepEqual({ ...vm.runInContext('deeperCellStats(7, 2)', p.context) }, { assigned: 6, kept: countedOf(2)[6] });
  p.socket().message({ ev: 'run', run: { ...state.run, proseg_counts: { reason: null, newest: 2, recent: [1, 2], sampled: [1, 2] } } }); await settle(20);
  assert.equal(vm.runInContext('frameSampled(2)', p.context), true);
  assert.equal(hover(), "12 µm² · 4 in proseg's count (as saved) · 2 set aside as background · 9 voxels · Unanalysed", 'the saved count at once');
  assert.equal(cardLine().title, READOUT_TIPS.saved);
  assert.deepEqual({ ...vm.runInContext('deeperCellStats(7, 2)', p.context) }, { assigned: 6, kept: 4 }, 'Go deeper\'s basis: Σf');
  assert.deepEqual(p.calls.counted, [2], 'nothing asked again');
  // Once heard of, it stays known when it has left the run's lists; the watch's counted event names them too.
  p.socket().message({ ev: 'run', run: { ...state.run, proseg_counts: { reason: null, newest: 90, recent: [90], sampled: [] } } }); await settle(20);
  assert.equal(vm.runInContext('frameSampled(2)', p.context), true);
  assert.equal(vm.runInContext('frameSampled(3)', p.context), false);
  p.socket().message({ ev: 'counted', run: 'r2-not-this-one', seqs: [3], sampled: [3] }); await settle(20);
  assert.equal(vm.runInContext('frameSampled(3)', p.context), false, "another run's list says nothing of this one");
  p.socket().message({ ev: 'counted', run: 'r1', seqs: [3], sampled: [3, 'x'] }); await settle(20);
  assert.equal(vm.runInContext('frameSampled(3)', p.context), true);
  // Go deeper reads the same (P3-1: an answer it holds on a frame's averaged count is asked for again once the frame
  // reads the count proseg saves there).
  assert.deepEqual([1, 2, 3, 4].map(seq => vm.runInContext(`deeperContext().sampled(${seq})`, p.context)), [true, true, true, false]);
  // Another run opened knows none of it.
  p.context.api.run = async id => ({ ...state.run, id, proseg_counts: { reason: null, newest: 5, recent: [5], sampled: [5] } });
  await vm.runInContext("openRun('r2')", p.context); await settle(20);
  assert.deepEqual([2, 3, 5].map(seq => vm.runInContext(`frameSampled(${seq})`, p.context)), [false, false, true]);
});

test("Go deeper's camera move carries its 3-D view to the stage's fit; its context names the run's buttons as they work; the stage's note names Start only while it works", async () => {
  const p = page();
  await settle(20);
  const { state, stage } = p.main(), fits = [];
  stage.fit = (...args) => fits.push(args);
  // onFly(rect, animate, view): view {box (µm), free (CSS px)} in 3-D, null in 2-D; the stage frames the box in the free
  // part (stage.fit's fourth argument).
  const view = { box: [0, 0, 120, 60], free: [440, 0, 1000, 800] };
  vm.runInContext(`deeperFly({ x0: 1, y0: 2, w: 3, h: 4 }, false, ${JSON.stringify(view)}); deeperFly({ x0: 5, y0: 6, w: 7, h: 8 }, true, null);`, p.context);
  assert.deepEqual(JSON.parse(JSON.stringify(fits)), [[{ x0: 1, y0: 2, w: 3, h: 4 }, false, 0, view], [{ x0: 5, y0: 6, w: 7, h: 8 }, true, 0, null]]);
  assert.equal(typeof p.calls.deeper.options.onFly, 'function');
  p.calls.deeper.options.onFly({ x0: 9, y0: 9, w: 1, h: 1 }, false, view);
  assert.deepEqual(fits.at(-1), [{ x0: 9, y0: 9, w: 1, h: 1 }, false, 0, view], 'the window\'s view reaches the stage');
  // The window's context: the run buttons that work now (run-buttons.mjs), as the run page draws them.
  const buttons = () => ({ ...vm.runInContext('deeperContext().buttons', p.context) });
  assert.deepEqual(buttons(), { start: false, resume: false, step: false, pause: true }, 'running: Pause');
  state.run = { ...state.run, state: 'paused', ack: { state: 'paused' } }; state.frames = state.frames.slice(0, 1); state.current = 0;
  assert.deepEqual(buttons(), { start: true, resume: false, step: true, pause: false }, 'on its prior: Start and Step');
  // The stage's note at frame 0 names Start only while a press of it works (the finish prompt open: none works).
  const note = () => vm.runInContext('stageNote(state.run)', p.context);
  assert.equal(note(), "This is proseg's starting point, built from the chosen prior. Press Start to begin sampling.");
  vm.runInContext('finishingPrompt = true', p.context);
  assert.equal(note(), "This is proseg's starting point, built from the chosen prior.");
  assert.deepEqual(buttons(), { start: false, resume: false, step: false, pause: false });
  vm.runInContext('finishingPrompt = false', p.context);
  state.run = { ...state.run, state: 'paused', ack: { state: 'running' } };
  assert.equal(note(), '', 'pausing: not the paused note');
});

test("the knob guide's timing line and Try message follow the run's buttons: written again when a press starts or stops working (the finish prompt), not on every render", async () => {
  // S-UIP-3: renderRun asks the live guide to write its lines again only when the buttons change; the guide reads them
  // from main.js (runButtons with the finish prompt), so its words name Resume or Step only while they work.
  const p = page();
  await settle(20);
  const { state } = p.main(), guide = vm.runInContext('knobs.guide', p.context);
  state.run = { ...state.run, state: 'paused', ack: { state: 'paused' } }; vm.runInContext('renderRun()', p.context);
  assert.deepEqual(plain(guide.options.buttons()), { start: false, resume: true, step: true, pause: false });
  const updates = guide.updates;
  vm.runInContext('renderRun(); renderRun()', p.context);
  assert.equal(guide.updates, updates, 'the same buttons: nothing written again');
  vm.runInContext('finishingPrompt = true; renderRun()', p.context);
  assert.equal(guide.updates, updates + 1, 'the finish prompt open: written again');
  assert.deepEqual(plain(guide.options.buttons()), { start: false, resume: false, step: false, pause: false });
  vm.runInContext('finishingPrompt = false; renderRun()', p.context);
  assert.equal(guide.updates, updates + 2, 'and again once it closes');
  assert.deepEqual(plain(guide.options.buttons()), { start: false, resume: true, step: true, pause: false });
});

test("the constants a run keeps from before proseg's count reach the clusters' settings and Compare, each run its own", async () => {
  // E4 (S-UIP-3): run.constants.kept names them (on a whole sample min_counts, the clusters' fewest transcripts, and
  // like_depth, Compare's like-for-like cut): the settings' tooltip and Compare's cut say they are not re-measured.
  const kept = ['glow', 'where_words', 'start_band', 'min_counts', 'like_depth'];
  const p = page({ realAnalysis: true, runOverride: { constants: { mode: 'whole', id: 'outline-2026-10', measured: false, kept } },
    routes: url => /\/runs\/[^/]+\/analysis$/.test(url) ? { ok: true, json: async () => ({ available: true, enabled: true, seqs: [] }) } : null });
  await settle(20);
  const { analysis } = p.main();
  assert.deepEqual(plain(analysis.settings.kept), kept, "the clusters' settings");
  await p.$('compare-button').click();
  assert.deepEqual(plain(p.calls.compare.opens.at(-1).kept), kept, "Compare's");
  // Another run (a region: none kept), then one that names no constants (an older engine's): nothing said.
  for (const [constants, expected] of [[{ mode: 'region', id: 'proseg-2026-10-31', measured: true, kept: [] }, []], [undefined, null]]) {
    p.context.api.run = async id => ({ ...p.main().state.run, id, constants });
    await vm.runInContext(`openRun('${expected ? 'r2' : 'r3'}')`, p.context); await settle(20);
    assert.deepEqual(plain(analysis.settings.kept), [], String(expected));
    await p.$('compare-button').click();
    assert.deepEqual(plain(p.calls.compare.opens.at(-1).kept), expected);
  }
});

test("the per-bin layer is asked for once per result and count: a frame counted again for the hold asks again, nothing else does", async () => {
  // A re-count that is not live keeps the seq's grid; a live one (a check switched on at the run's end) can write
  // another: the page asks once more when the applied summary's count changes (critic R2-missed-caches).
  const p = page();
  await settle(20);
  const asked = [];
  p.context.fetchBins = async (run, seq) => { asked.push(seq); return null; };
  vm.runInContext('qcOverlay.wantsBins = () => true', p.context);
  const apply = (seq, count, summary = true) => vm.runInContext(`qcClient.view = { atStart: false }; qcClient.applied = { seq: ${seq}, point: { count: '${count}' },
    summary: ${summary ? `{ count: '${count}' }` : 'null'} }; qcBins(); qcBins();`, p.context);
  apply(1, 'averaged'); await settle();
  assert.deepEqual(asked, [1], 'once per result');
  apply(1, 'averaged', false); await settle();
  assert.deepEqual(asked, [1], 'its point alone (its summary on its way): the same count, not asked again');
  apply(1, 'sampled'); await settle();
  assert.deepEqual(asked, [1, 1], 'counted again for the hold: asked again');
  apply(1, 'sampled'); await settle();
  assert.deepEqual(asked, [1, 1]);
  apply(2, 'averaged'); await settle();
  assert.deepEqual(asked, [1, 1, 2]);
});

test('sparklines follow their canvas when its width changes without a window resize', async () => {
  const p = page();
  await settle(20);
  p.select(6); await settle(20);
  const canvases = p.$('cell-card').all().filter(n => n.tagName === 'CANVAS'), ro = p.observers.find(o => canvases.every(c => o.nodes.has(c)));
  assert.equal(canvases.length, 2); assert.deepEqual([...ro.nodes], canvases);
  assert.deepEqual(canvases.map(c => [c.width, c.height]), [[200, 56], [200, 56]]);
  // The side column's scrollbar goes away (a panel was collapsed): the browser reports the canvases' new size.
  ro.fn(canvases.map(target => ({ target, contentRect: { width: 215, height: 56 } })));
  assert.deepEqual(canvases.map(c => c.width), [215, 215]); assert.deepEqual([...canvases[0].size], [215, 56]);
  ro.fn(canvases.map(target => ({ target, contentRect: { width: 0, height: 0 } })));   // collapsed: nothing to draw
  assert.deepEqual(canvases.map(c => c.width), [215, 215]);
  vm.runInContext('state.selected = 0; updateCellCard();', p.context);
  assert.equal(ro.nodes.size, 0, 'a released cell is no longer observed');
});

test('each streamed snapshot is painted once: the HUD, status line, facts and active cells change in one batch', async () => {
  const p = page();
  await settle(20);
  const { state, paint } = p.main(), batches = [];
  for (let seq = 2; seq < 6; seq++) {
    const before = paint.flushes;
    p.socket().message({ ev: 'frames', run: 'r1', from: seq, frames: [p.frame(seq)] });
    // Until the new frame is uploaded nothing is written: the status line still names the frame on screen.
    assert.match(p.$('status').textContent, new RegExp(`^Iteration ${seq - 1} · `));
    await settle(20);
    batches.push(paint.flushes - before);
    assert.equal(p.$('hud-iter').textContent, String(seq)); assert.match(p.$('status').textContent, new RegExp(`^Iteration ${seq} · `));
    assert.match(p.$('facts').textContent, new RegExp(`Iteration${seq}`));
  }
  assert.deepEqual(batches, [1, 1, 1, 1], 'one batch (one paint) per snapshot');
  assert.equal(state.current, 5); assert.equal(paint.pending, false);
});

test('Clusters and the UMAP found unable to run mid-session: /health is asked again, and the mixing check says why', async () => {
  // The engine's worker said its Python cannot run them (a run's analysis pushed "unavailable"): the page asks /health
  // once more, so the next run's pane stays shut (engineCan) and the mixing check, which scores nothing without them,
  // says so in its panel and chip (qc-panel.js withEngine), switch on or off.
  let health = { proseg: true };
  const asked = [];
  const p = page({ apiOverride: { health: async () => { asked.push(health); return health; } } });
  await settle(20);
  assert.equal(p.calls.panel.at(-1).state, 'live'); assert.equal(asked.length, 1);
  health = { proseg: true, analysis: false, qc: { available: false, state: 'ready', reason: 'analysis' } };
  p.socket().message({ ev: 'analysis', run: 'r1', state: 'unavailable', enabled: true });
  p.socket().message({ ev: 'analysis', run: 'r1', state: 'unavailable', enabled: true });   // one question at a time
  await settle(20);
  const { state, analysis } = p.main();
  assert.equal(asked.length, 2); assert.equal(state.health.qc.reason, 'analysis'); assert.equal(analysis.engineCan, false);
  assert.equal(p.calls.panel.at(-1).state, 'unavailable'); assert.equal(p.calls.chip.at(-1).state, 'unavailable');
});

test("an old proseg-live: how /health says to update it reaches the mixing check's line and the clusters' line, and the next run's after /health is asked again", async () => {
  // Both describes say the run has no proseg's count: its proseg-live can't keep it (old_proseg). GET /health
  // proseg_fix (how this app finds the program) says how to update it, in each line's tooltip.
  let health = { proseg: true, proseg_fix: 'install' };
  const p = page({ realPanel: true, realAnalysis: true, qc: qcEngine({ state: 'unavailable', reason: 'old_proseg' }), apiOverride: { health: async () => health },
    routes: url => /\/runs\/[^/]+\/analysis$/.test(url) ? { ok: true, json: async () => ({ available: true, enabled: true, seqs: [], unavailable: 'old_proseg' }) } : null });
  p.context.fromView = realFromView;   // the panel's own state from the view: with its reason and the page's prosegFix
  await settle(20);
  const { analysis } = p.main(), line = () => vm.runInContext('qcPanel.status.childNodes[0]', p.context);
  assert.equal(line().textContent, QC_TEXT.unavailableOldProseg); assert.equal(line().title, oldProsegTooltip('install'));
  assert.equal(analysis.blocked, 'old_proseg'); assert.equal(analysis.note.hidden, false);
  assert.equal(analysis.note.textContent, 'Update proseg-live to see clusters.'); assert.equal(analysis.note.title, oldProsegTooltip('install'));
  assert.notEqual(oldProsegTooltip('install'), oldProsegTooltip('build'));
  // /health asked again (a run's analysis went "unavailable"): its newer word reaches the panel at once, and the line of
  // the next run that has no proseg's count.
  health = { proseg: true, proseg_fix: 'build' };
  p.socket().message({ ev: 'analysis', run: 'r1', state: 'unavailable', enabled: true }); await settle(20);
  assert.equal(line().title, oldProsegTooltip('build'));
  p.context.api.run = async id => ({ ...p.main().state.run, id });
  await vm.runInContext("openRun('r2')", p.context); await settle(20);
  assert.equal(p.main().state.run.id, 'r2'); assert.equal(analysis.blocked, 'old_proseg'); assert.equal(analysis.note.title, oldProsegTooltip('build'));
  assert.equal(line().title, oldProsegTooltip('build'));
  // Go deeper's window reads the same word, and in which session (a notebook's reads its own settings file).
  const context = p.calls.deeper.options.context();
  assert.deepEqual([context.prosegFix, context.notebook], ['build', false]);
});

test('telemetry describes the open run without cell ids or paths', async () => {
  const p = page();
  await settle(20);
  const { context, counters } = p.telemetry(), summary = context();
  assert.deepEqual(JSON.parse(JSON.stringify(summary)), { mode: 'detail', state: 'running', cells: null, live: true, umap: false, view: '2d', level: null, stage: [1000, 800], renderer: null });
  const counts = counters();
  assert.equal(counts.snapshots, 1); assert.equal(typeof counts.batches, 'number');
  assert.doesNotMatch(JSON.stringify(summary), /Sample|c6|\//);
});

test('the mixing check: a result is shown with the snapshot of its own frame, never on an earlier one; colours, tooltip, card and switch', async () => {
  const p = page();
  await settle(20);
  const { state, qcClient, paint, stage, layerPanel } = p.main();
  assert.deepEqual(p.qc.calls.slice(0, 1), ['describe'], 'the run\'s check is asked for when it opens');
  assert.equal(p.calls.panel.at(-1).state, 'live'); assert.equal(p.calls.panel.at(-1).summary, null, 'no result yet: waiting for one');
  assert.equal(layerPanel.mixingAvailable, true, 'Colour by: Mixing is offered while the check is on');
  assert.equal(p.calls.lane.at(-1).start, 10, 'the timeline lane shows the start line');

  // A result for frame 2 while frame 1 is on screen: never shown on the earlier frame.
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2, { areas: [{ L: 'A', tag: 'rose', per100: 23, outline: [[0, 0], [1, 0], [1, 1]] }] }) });
  await settle(20);
  assert.equal(state.current, 1); assert.equal(qcClient.applied, null);
  assert.equal(p.calls.panel.at(-1).summary, null);
  // Frame 2 arrives: its result is applied in the snapshot's own batch. Its areas stay in the view (the cards' numbers)
  // and never reach the overlay (none is drawn since 2026-10-22): the per-bin layer's level does.
  const flushes = paint.flushes;
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  assert.equal(state.current, 2); assert.equal(qcClient.applied.seq, 2);
  assert.equal(paint.flushes - flushes, 1, 'one batch: the result shows in the frame of the snapshot');
  assert.equal(p.calls.panel.at(-1).summary.shown, 11); assert.equal(p.calls.chip.at(-1).summary.shown, 11);
  assert.deepEqual(qcClient.view.areas.map(a => a.L), ['A']);
  const [level] = p.calls.levels.at(-1);
  assert.deepEqual([level.anchorPer100, level.tissuePer100], [10, 11], 'the start line\'s level, else the headline\'s own integer');
  p.socket().message({ ev: 'frames', run: 'r1', from: 3, frames: [p.frame(3)] });
  await settle(20);

  // Mixing colours (the panel's button, key M, Layers): the values load, and colour the next snapshot.
  vm.runInContext('setMixingColours(true)', p.context);
  assert.equal(stage.options.cellColorBy, 'mixing');
  await settle(20);
  p.socket().message({ ev: 'frames', run: 'r1', from: 4, frames: [p.frame(4)] });
  await settle(20);
  assert.ok(p.qc.calls.some(c => Array.isArray(c) && c[0] === 'cells' && c[1] === 2));
  assert.deepEqual([...p.calls.mixing.at(-1)], [10, 80, 255, 30, 0, 90, 76], 'the stage colours the cells with the result shown');
  // The cell under the pointer (label 7) says how mixed it looks; selected, the mixing check's row for it says so too,
  // against the start (the cell card keeps its name, readout and history).
  p.$('stage').emit('pointermove', { offsetX: 100, offsetY: 30, pointerId: 1, button: 0 });
  assert.match(p.$('tooltip').textContent, /Looks mixed · the model reads about 30 % of its RNA as coming from neighbouring cells \(estimate\)/);
  p.select(6); await settle(20);
  vm.runInContext('updateCellCard()', p.context); await settle(20); vm.runInContext('updateCellCard()', p.context);
  assert.match(p.calls.cells.at(-1).line, /^Mixing: the model reads about 30 % of its RNA as coming from neighbouring cells \(30 % at iteration 0\)\. A model's estimate, not proof of a boundary error\. Between cells of the same kind it can't tell RNA apart\./);
  assert.doesNotMatch(p.$('cell-card').textContent, /Mixing/);

  // The switch: off for this run; the cells go back to their colours, the values leave the stage.
  await vm.runInContext('qcClient.setEnabled(false)', p.context);
  await settle(20);
  assert.deepEqual(p.qc.calls.filter(c => Array.isArray(c) && c[0] === 'post').at(-1), ['post', { enabled: false }]);
  assert.equal(p.calls.panel.at(-1).state, 'off'); assert.equal(stage.options.cellColorBy, 'identity');
  assert.equal(p.calls.mixing.at(-1), null); assert.equal(layerPanel.mixingAvailable, false);
  assert.equal(p.calls.lane.at(-1), null, 'no lane while the check is off');
  assert.doesNotMatch(p.$('tooltip').textContent, /mixed|mixing/i);
  // The stage chip opens the panel (and scrolls to it).
  vm.runInContext('qcChip.onClick()', p.context);
  assert.equal(p.calls.panel.at(-1), 'open'); assert.equal(p.$('qc-panel').hidden, false);
});

test("the mixing check at frame 0 and while cells form: the per-bin layer's level is the start line's, no area reaches the overlay; key M says why it cannot colour", async () => {
  const p = page();
  await settle(20);
  const { state, stage } = p.main();
  const area = (L, tag) => ({ L, tag, per100: 18, start_per100: 18, judged: 400, outline: [[0, 0], [1, 0], [1, 1]] });
  p.socket().message({ ev: 'qc', run: 'r1', state: 'start', summary: { per100: 10, judged: 300, mixed: 30, median_tx: 386, areas: [area('A', 'already')], more_spots: 0 } });
  vm.runInContext('setLive(false); show(0)', p.context);
  await settle(20);
  assert.equal(state.current, 0);
  let [level] = p.calls.levels.at(-1);
  assert.equal(level.anchorPer100, 10, 'frame 0: the start line\'s level (its areas stay in the view, never drawn)');
  assert.equal(p.calls.autoPrior.at(-1), true, "frame 0 shows the prior's own layer by itself ...");
  vm.runInContext('setMixingColours(true)', p.context);
  assert.equal(stage.options.cellColorBy, 'mixing'); assert.equal(p.calls.autoPrior.at(-1), false, '... but not over the Mixing colours');
  vm.runInContext('setMixingColours(false)', p.context); assert.equal(p.calls.autoPrior.at(-1), true);
  // A frame while cells still form (area_view "start"), then from burn-in + 20 iterations the frame's own areas: the
  // overlay gets the level all along, never an area.
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 1, iter: 1, summary: qcSummary(1, { phase: 'burnin', areas: [], area_view: 'start' }) });
  vm.runInContext('show(1)', p.context);
  await settle(20);
  assert.equal(state.current, 1); assert.equal(p.calls.panel.at(-1).summary.seq, 1);
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2, { areas: [area('B', 'rose')], area_view: 'smoothed' }) });
  vm.runInContext('show(2)', p.context);
  await settle(20);
  [level] = p.calls.levels.at(-1);
  assert.deepEqual([level.anchorPer100, level.tissuePer100], [10, 11]);
  // The check switched off: key M colours nothing and says why, in plain words.
  await vm.runInContext('qcClient.setEnabled(false)', p.context);
  await settle(20);
  for (const fn of p.context.document.listeners.get('keydown')) fn({ key: 'm', target: p.$('stage'), preventDefault() {} });
  assert.notEqual(stage.options.cellColorBy, 'mixing');
  assert.equal(p.$('notice').hidden, false); assert.equal(p.$('notice').textContent, QC_TEXT.mixingOffTooltip);
});

test('the mixing check: paused, a result that arrives for the frame on screen lands without motion; following live it may move', async () => {
  const p = page();
  await settle(20);
  const { state, qcClient } = p.main();
  const area = L => ({ L, tag: 'rose', per100: 20, start_per100: 10, judged: 400, outline: [[0, 0], [1, 0], [1, 1]] });
  vm.runInContext('setMixingColours(true)', p.context);
  // Following live: the cells may cross-fade (the stage's own 1 s rule decides); the first areas land at once (nothing
  // on screen before them to move from).
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 1, iter: 1, summary: qcSummary(1, { areas: [area('A')] }) });
  await settle(20); vm.runInContext('qcClient.frame()', p.context); await settle(20); vm.runInContext('qcClient.frame()', p.context);
  assert.equal(p.calls.mixingOptions.motion, true);
  // Paused on frame 1: a late result for it (its values fetched again) lands at once, in the overlay and on the cells.
  state.run.state = 'paused'; state.run.ack = { state: 'paused' }; vm.runInContext('setLive(false)', p.context);
  qcClient.cells.delete(1);
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 1, iter: 1, summary: qcSummary(1, { areas: [area('A'), area('B')], per100: 12, shown: 12 }) });
  await settle(20);
  assert.equal(qcClient.view.still, true);
  assert.equal(p.calls.levels.at(-1)[0].tissuePer100, 12, 'the overlay got its level (never its areas)');
  assert.ok(p.calls.mixing.at(-1) instanceof Uint8Array); assert.equal(p.calls.mixingOptions.motion, false, 'so did the cells');
  assert.equal(p.calls.levels.at(-1)[1], false, 'the calm switch is never touched'); assert.equal(p.calls.mixingCalm, false);
});

test('the mixing check: a switch flipped in another window reaches this page; the stage never changes size for the check', async () => {
  const p = page();
  await settle(20);
  const { stage, layerPanel } = p.main();
  assert.equal(p.calls.panel.at(-1).state, 'live'); assert.equal(layerPanel.mixingAvailable, true);
  assert.doesNotMatch(source, /with-qc|classList\.toggle\('with/, 'the lane is drawn inside the timeline: the dock (and the stage) keeps its size');
  vm.runInContext('setMixingColours(true)', p.context); assert.equal(stage.options.cellColorBy, 'mixing');
  p.socket().message({ ev: 'qc', run: 'r1', state: 'switch', enabled: false });
  await settle(20);
  assert.equal(p.calls.panel.at(-1).state, 'off'); assert.equal(stage.options.cellColorBy, 'identity', 'the cells go back to their colours');
  assert.equal(p.calls.lane.at(-1), null); assert.equal(layerPanel.mixingAvailable, false);
  const asked = p.qc.calls.length;
  p.socket().message({ ev: 'qc', run: 'r1', state: 'switch', enabled: true });
  await settle(20);
  assert.ok(p.qc.calls.slice(asked).includes('describe'), 'switched on elsewhere: the check is asked for again');
  assert.equal(p.calls.panel.at(-1).state, 'live'); assert.equal(layerPanel.mixingAvailable, true);
});


test('the display modes are Cells (key 1) and Outlines (key 2), nothing else; a name that is no mode any more shows Cells', async () => {
  const p = page();
  await settle(20);
  const { stage, state } = p.main();
  const key = k => { for (const fn of p.context.document.listeners.get('keydown')) fn({ key: k, target: p.$('stage'), preventDefault() {} }); };
  const pressed = () => p.$('mode').children.filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.value);
  assert.deepEqual(p.$('mode').children.map(b => b.dataset.value), ['cells', 'outlines'], 'the buttons (index.html): no Activity');
  assert.equal(stage.options.mode, 0); assert.deepEqual(pressed(), ['cells']);
  key('2'); assert.equal(stage.options.mode, 1); assert.equal(state.display.mode, 'outlines'); assert.deepEqual(pressed(), ['outlines']);
  key('3'); assert.equal(stage.options.mode, 1, '3 is no mode any more'); assert.deepEqual(pressed(), ['outlines']);
  key('1'); assert.equal(stage.options.mode, 0); assert.deepEqual(pressed(), ['cells']);
  p.$('mode').emit('click', { target: p.$('mode').children[1] }); assert.equal(stage.options.mode, 1); assert.deepEqual(pressed(), ['outlines']);
  vm.runInContext("setMode('activity')", p.context);
  assert.equal(stage.options.mode, 0); assert.equal(state.display.mode, 'cells'); assert.deepEqual(pressed(), ['cells']);
  assert.doesNotMatch(source, /activity/i, 'nothing on the page knows the Activity mode');
});

test('the mixing check is off by default: Mixing is not offered, and the picker says how a sample\'s check stands only once it is on for new runs', async () => {
  // The run's check is off (as every new run's is until a user turns it on); the engine remembers what this page posts.
  const qc = qcEngine({ enabled: false, state: 'off', start: null }), post = qc.api.setRunQc;
  qc.api.setRunQc = async (id, body) => { if (body.enabled !== undefined) qc.describe.enabled = body.enabled; return post(id, body); };
  const p = page({ qc });
  await settle(20);
  const { state, layerPanel } = p.main();
  assert.equal(p.calls.panel.at(-1).state, 'off'); assert.equal(p.calls.panel.at(-1).enabled, false);
  assert.equal(layerPanel.mixingAvailable, false); assert.equal(layerPanel.mixingWhy, null, 'Colour by: Mixing is not listed (no reason to give)');
  assert.equal(p.calls.lane.at(-1), null); assert.equal(p.calls.mixing.length, 0, 'nothing colours the cells');
  const line = () => p.$('picker-qc').hidden ? '' : p.$('picker-qc').textContent;
  const pick = async () => {
    vm.runInContext('enterPick("s")', p.context);
    for (let i = 0; i < 40 && state.mode !== 'pick'; i++) await settle(1);
    await settle(20); assert.equal(state.mode, 'pick');
  };
  const learning = { ev: 'qc', sample: 's', state: 'learning', progress: { step: 3, of: 50 }, eta_s: 90 };
  await pick();
  assert.equal(line(), '', 'a sample learned before says nothing while the check is off');
  p.socket().message(learning); assert.equal(line(), '', 'never "learning this sample now" while the check is off');
  // Switched on elsewhere while this page watched a run (the engine starts new runs with it from then on).
  await vm.runInContext('openRun("r1")', p.context); await settle(20);
  assert.equal(p.calls.panel.at(-1).state, 'off');
  p.socket().message({ ev: 'qc', run: 'r1', state: 'switch', enabled: true, n: 5 }); await settle(20);
  await pick();
  assert.equal(line(), QC_TEXT.pickerReady);
  p.socket().message(learning); assert.equal(line(), copy('pickerLearning', { t: duration(90) }));
  p.socket().message({ ev: 'qc', sample: 's', state: 'waiting', reason: 'gpu_busy' }); assert.equal(line(), QC_TEXT.pickerWaiting);
  p.socket().message({ ev: 'qc', sample: 'other', state: 'learning' }); assert.equal(line(), QC_TEXT.pickerWaiting, 'another sample\'s says nothing here');
  // Opening a run is no flip (its switch may be an old one: this one still says off); this page's own switch is.
  await vm.runInContext('openRun("r1")', p.context); await settle(20);
  assert.equal(p.calls.panel.at(-1).state, 'off');
  await pick(); assert.equal(line(), QC_TEXT.pickerReady);
  await vm.runInContext('openRun("r1")', p.context); await settle(20);
  await vm.runInContext('qcClient.setEnabled(true)', p.context); await settle(20);
  assert.notEqual(p.calls.panel.at(-1).state, 'off');
  await vm.runInContext('qcClient.setEnabled(false)', p.context); await settle(20);
  assert.equal(p.calls.panel.at(-1).state, 'off');
  assert.deepEqual(p.qc.calls.filter(c => Array.isArray(c) && c[0] === 'post').map(c => c[1]), [{ enabled: true }, { enabled: false }]);
  await pick(); assert.equal(line(), '', 'switched off again: nothing again');
  p.socket().message(learning); assert.equal(line(), '');
});

test('a freshly loaded page learns what new runs start with from the engine\'s answer about the sample (new_runs): its newer word wins', async () => {
  // The user switched the check on in an earlier session: the engine remembers it (prefs.json) and says so; this page
  // has seen no switch of its own.
  let remembered = true, learned = 'ready';
  const qc = qcEngine({ enabled: true });
  qc.api.sampleQc = async () => ({ state: learned, new_runs: remembered });
  const p = page({ qc });
  await settle(20);
  const { state } = p.main();
  const line = () => p.$('picker-qc').hidden ? '' : p.$('picker-qc').textContent;
  const pick = async () => {
    vm.runInContext('enterPick("s")', p.context);
    for (let i = 0; i < 40 && state.mode !== 'pick'; i++) await settle(1);
    await settle(20); assert.equal(state.mode, 'pick');
  };
  await pick();
  assert.equal(line(), QC_TEXT.pickerReady, 'on for new runs (the engine\'s remembered choice): the line shows');
  p.socket().message({ ev: 'qc', sample: 's', state: 'learning', eta_s: 90 });
  assert.equal(line(), copy('pickerLearning', { t: duration(90) }), 'and follows the sample\'s events');
  remembered = false;                         // switched off since (another window)
  await pick();
  assert.equal(line(), '', 'the engine\'s newer answer wins: nothing while off');
  // On for new runs, on a sample never learned: Start will learn it first (minutes of GPU 0), said before Start (it
  // used to say nothing, and the fit queued at Start without warning).
  remembered = true; learned = 'off';
  await pick();
  assert.equal(line(), QC_TEXT.pickerLearnFirst);
  assert.equal(QC_TEXT.pickerLearnFirst, 'Mixing check: on. Starting learns this sample first (a few minutes, once).');
  remembered = false;
  await pick();
  assert.equal(line(), '', 'off for new runs: nothing, learned or not');
});

test('held at the end of proseg\'s standard run, the stage\'s note gives way to the test corner while it is open', async () => {
  // Both sit at the stage's bottom left, and a test comes only at the end of the run, where the note shows.
  const p = page();
  await settle(20);
  const { state } = p.main(), corner = p.calls.corner, note = p.$('prior-hint');
  const held = { ...state.run, state: 'paused', schedule_done: { iter: 1, seq: 1 }, ack: { state: 'paused', iter: 1, seq: 1 } };
  p.socket().message({ ev: 'run', run: held }); await settle(20);
  assert.equal(note.hidden, false); assert.match(note.textContent, /^Proseg's standard run is complete/);
  corner.options.onBox({ left: 12, top: 640, bottom: 12, w: 360, h: 248 }); await settle(20);
  assert.equal(note.hidden, true, 'hidden under the open corner');
  corner.options.onBox(null); await settle(20);
  assert.equal(note.hidden, false, 'back once the corner closed'); assert.match(note.textContent, /^Proseg's standard run is complete/);
});
test('held at the end of proseg\'s standard run, Space and S do not leave it: the note stands out and asks for a click', async () => {
  // Resume there gives up proseg's own result for good (it samples on at temperature 1): a stray key must not do it;
  // the buttons still work, and Finish & save is the main one there.
  const p = page();
  await settle(20);
  const { state } = p.main();
  const held = { ...state.run, state: 'paused', schedule_done: { iter: 1, seq: 1 }, ack: { state: 'paused', iter: 1, seq: 1 } };
  p.socket().message({ ev: 'run', run: held }); await settle(20);
  assert.match(p.$('prior-hint').textContent, /^Proseg's standard run is complete\. Finish & save keeps this result\. Resume samples on/);
  const classes = new Set();
  p.$('prior-hint').classList = { add: c => classes.add(c), remove: c => classes.delete(c), toggle() {} };
  const keydown = p.context.document.listeners?.get?.('keydown') ?? [];
  const press = key => { for (const fn of keydown) fn({ key, target: { closest: () => null }, preventDefault() {}, metaKey: false, ctrlKey: false, altKey: false }); };
  const sent = p.calls.controls.length;
  press(' '); press('s'); await settle(5);
  assert.equal(p.calls.controls.length, sent, 'no resume, no step from a key');
  assert.ok(classes.has('attention'), 'the note stands out instead');
  const logged = [];
  p.context.sopaCompanionLog = { add: (kind, text) => logged.push(`${kind}: ${text}`) };
  p.$('run-toggle').click(); await settle(5);
  assert.deepEqual(p.calls.controls.slice(sent), ['resume'], 'the button itself still resumes');
  assert.deepEqual(logged, ['action: resume (at iteration 1)'], 'the log has the user\'s action');
  // On its prior (one snapshot, never sampled) the same button says Start, and so does the log.
  state.frames.splice(1); state.current = 0;
  p.socket().message({ ev: 'run', run: { ...state.run, state: 'paused', frames: 1, schedule_done: undefined, ack: { state: 'paused', iter: 0, seq: 0 } } });
  await settle(20);
  assert.equal(p.$('run-toggle').textContent, 'Start');
  p.$('run-toggle').click(); await settle(5);
  assert.equal(logged.at(-1), 'action: start (at iteration 0)');
});

test('Mixing colours kept from an earlier session, on a run whose check is off: the clusters\' colours; a colour chosen on this page comes back', async () => {
  const p = page({ qc: qcEngine({ enabled: false, state: 'off', start: null }) });
  await settle(20);
  const { stage, layerPanel } = p.main();
  assert.equal(p.calls.panel.at(-1).state, 'off'); assert.equal(layerPanel.mixingAvailable, false);
  // The saved settings bring Mixing back (an earlier session's choice): this page has seen no other colour, so the cells
  // take the clusters' colours, which every run has now (clusters and UMAP are always on), not their identity colours.
  stage.set({ cellColorBy: 'mixing', transcriptColorBy: 'mixing' });
  vm.runInContext('qcClient.changed(true)', p.context); await settle(20);
  assert.equal(stage.options.cellColorBy, 'cluster'); assert.equal(stage.options.transcriptColorBy, 'cell');
  assert.equal(p.calls.mixing.length, 0, 'nothing colours the cells by mixing');
  // A colour chosen here (Layers > Cells > Colour by) is the one the cells go back to.
  vm.runInContext("colourBy('identity')", p.context); assert.equal(stage.options.cellColorBy, 'identity');
  stage.set({ cellColorBy: 'mixing' });
  vm.runInContext('qcClient.changed(true)', p.context); await settle(20);
  assert.equal(stage.options.cellColorBy, 'identity');
  // Key M while the check is off: nothing changes, and the notice says how to get Mixing.
  for (const fn of p.context.document.listeners.get('keydown')) fn({ key: 'm', target: p.$('stage'), preventDefault() {} });
  assert.equal(stage.options.cellColorBy, 'identity'); assert.equal(p.$('notice').textContent, QC_TEXT.mixingOffTooltip);
});

test('2026-10-22, no flagged area on screen: flagged areas never reach the stage: no area handed to the overlay, no area line in the tooltip, a click selects the cell under it', async () => {
  const p = page({ hook: true });
  await settle(20);
  const { state } = p.main(), canvas = p.$('stage');
  const area = (L, x) => ({ L, tag: 'rose', per100: 20, start_per100: 10, judged: 400, outline: [[x, 0], [x + 50, 0], [x + 50, 50], [x, 50]] });
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2, { areas: [area('A', 0), area('B', 100)] }) });
  await settle(20); vm.runInContext('qcClient.frame()', p.context); await settle(20);
  assert.equal(state.current, 2);
  assert.deepEqual(vm.runInContext('qcClient.view.areas.map(a => a.L)', p.context), ['A', 'B'], 'the engine\'s areas stay in the view, for the cards');
  // main.js has no way left to select, hover or draw an area (the overlay's mock has none of those either: a call would throw).
  for (const name of ['selectArea', 'qcObstacles', 'qcSelected', 'qcAreasKey']) assert.equal(vm.runInContext(`typeof ${name}`, p.context), 'undefined', name);
  assert.ok(p.calls.levels.length > 0, 'the overlay gets the per-bin layer\'s level');
  // The pointer over area A's place: the cell's lines, no area line.
  p.$('stage').emit('pointermove', { offsetX: 10, offsetY: 10, pointerId: 1, button: 0 });
  const lines = p.$('tooltip').children.map(n => n.textContent);
  assert.ok(lines.length >= 2, lines.join(' | ')); assert.ok(lines.every(l => !/\barea\b/i.test(l)), lines.join(' | '));
  // A click there (pressed and let go without moving) selects the cell under it: no chip to catch it.
  Object.assign(canvas, { setPointerCapture() {}, hasPointerCapture: () => true, releasePointerCapture() {} });
  for (const type of ['pointerdown', 'pointerup']) canvas.emit(type, { offsetX: 10, offsetY: 10, pointerId: 1, button: 0, shiftKey: false });
  assert.equal(state.selected, 7);
  canvas.emit('pointerleave', {});
  // The harness: the result's letters (the engine's, checked against the frame on screen) and none drawn; no word of an
  // area drawn on a canvas.
  const snap = p.snapshot();
  assert.deepEqual(snap.qc.areas, ['A', 'B']); assert.equal(snap.qc.drawn, 0);
  assert.ok(vm.runInContext('window.sopaCompanion.qcTexts()', p.context).every(t => !/^[A-H]( · |$)/.test(t)));
});

test('the mixing check: paused on the newest frame with Mixing colours, its late result never greys the cells in between', async () => {
  const p = page();
  await settle(20);
  const { state, qcClient } = p.main();
  vm.runInContext('setMixingColours(true)', p.context);
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 1, iter: 1, summary: qcSummary(1) });
  await settle(20); vm.runInContext('qcClient.frame()', p.context); await settle(20);
  assert.ok(p.calls.mixing.at(-1) instanceof Uint8Array, 'frame 1 coloured');
  state.run.state = 'paused'; state.run.ack = { state: 'paused' }; vm.runInContext('setLive(false)', p.context);
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] }); await settle(20);
  vm.runInContext('show(2)', p.context); await settle(20);
  const before = p.calls.mixing.length;
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2) });   // the paused frame's own
  await settle(30);
  const sequence = p.calls.mixing.slice(before);
  assert.ok(sequence.length >= 1 && sequence.every(bytes => bytes instanceof Uint8Array), `every setMixing call has values: ${sequence.map(b => b && 'bytes')}`);
  assert.ok(!p.calls.mixing.includes(null), 'never null (grey) between two results');
});

// The card's actions through the page.
const plain = value => JSON.parse(JSON.stringify(value));   // objects made inside the page's context (another realm)
const r4Card = { id: 'R4:cell-compactness:0.02', rule: 'R4', areas: ['A'], flag: 'cell-compactness', from: 0.04, to: 0.02, confidence: 'rough', timing: 'early',
  actions: ['try', 'new_run', 'why'], main: 'try', growth: 0.18 };

test('the mixing check: Try it sets the knob through the Live knobs session, then registers the trial; one click at a time; Undo puts it back', async () => {
  const p = page();
  await settle(20);
  const { qcClient } = p.main();
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 1, iter: 1, summary: qcSummary(1, { suggestion: r4Card }) });
  await settle(20); vm.runInContext('qcClient.frame()', p.context); await settle(20);
  const panel = vm.runInContext('qcPanel', p.context);
  assert.equal(panel.options.suggestions, true, "the slot is on"); assert.equal(typeof panel.options.knobLabel, 'function');
  assert.equal(p.calls.panel.at(-1).suggestion.rule, 'R4');
  // A double click: one change, one trial.
  const shown = p.calls.panel.length, first = panel.options.onTry(r4Card), second = panel.options.onTry(r4Card);
  assert.equal(p.calls.panel.at(-1).acting, true, 'every action waits while one is on its way (the card\'s buttons too)');
  await Promise.all([first, second]); await settle(20);
  assert.ok(p.calls.panel.slice(shown).some(s => s.acting === true));
  assert.deepEqual(plain(p.calls.tries), [['cell-compactness', 0.02, 0.04]]);
  assert.deepEqual(plain(p.qc.calls.filter(c => c[0] === 'trial')), [['trial', { rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02 }]]);
  assert.equal(p.calls.panel.at(-1).acting, false); assert.equal(p.calls.panel.at(-1).trial.id, 1);
  assert.equal(qcClient.trial.id, 1);
  // Undo: the trial's old value back through the same session, then the trial ends as undone.
  await panel.options.onUndo(qcClient.trial); await settle(20);
  assert.deepEqual(plain(p.calls.undos), [['cell-compactness', 0.04]]); assert.deepEqual(plain(p.qc.calls.at(-1)), ['end', 'undone', null]);
  assert.equal(p.calls.panel.at(-1).trial, null); assert.equal(p.calls.panel.at(-1).trialEnded.outcome, 'undone');
  // A refused change says why in the session's own words, and registers nothing.
  vm.runInContext('knobs.refuse = "Already at this value."', p.context);
  await panel.options.onTry(r4Card); await settle(20);
  assert.equal(p.$('notice').textContent, QC_TEXT.tryAlready, "the session's reason in the deck's words");
  assert.equal(p.qc.calls.filter(c => c[0] === 'trial').length, 1);
  vm.runInContext('knobs.refuse = "Failed to fetch"', p.context);
  await panel.options.onTry(r4Card); await settle(20);
  assert.equal(p.$('notice').textContent, QC_TEXT.tryFailed, "the browser's own words never reach the page");
  vm.runInContext('knobs.refuse = null', p.context);
  // The Undo card: Undo it puts the old value back and answers it; Keep only answers it.
  const undo = { id: 'U:cell-compactness:0.04', rule: 'U', areas: [], flag: 'cell-compactness', from: 0.02, to: 0.04, confidence: 'seen' };
  await panel.options.onUndo(undo); await settle(20);
  assert.deepEqual(plain(p.calls.undos.at(-1)), ['cell-compactness', 0.04]); assert.deepEqual(plain(p.qc.calls.at(-1)), ['end', 'undone', 'U']);
  await panel.options.onKeep({ ...undo, to: 0.05 }); await settle(20);
  assert.deepEqual(plain(p.qc.calls.at(-1)), ['end', 'kept', 'U']); assert.equal(p.calls.undos.length, 2, 'Keep changes nothing');
});

test('the mixing check: New run with this setting leaves this run and fills the picker with its region, starting cells and options plus the setting', async () => {
  const p = page();
  await settle(20);
  const { state } = p.main();
  // A run made with the old schedule presets names one: the new run follows proseg's own schedule whatever it was.
  Object.assign(state.run, { region: { kind: 'box', box: [10, 20, 310, 320], bbox: [10, 20, 310, 320] }, prior: { cells: { kind: 'column', column: 'other' }, nuclei: null },
    schedule: 'full', options: { 'voxel-layers': 3 } });
  state.health = { ...state.health, proseg: true };
  state.runs = [{ ...state.run }];
  const panel = vm.runInContext('qcPanel', p.context);
  const asked = panel.options.onNewRun(r4Card);
  await discardRun(p);
  // Filled in the step that first lets Start go ahead (it is never greyed out: until then a press says the sample is still
  // loading): never a moment with the default region, while the overview loads.
  const ready = () => vm.runInContext('pickerProblems().length === 0', p.context);
  for (let i = 0; i < 40 && state.mode !== 'pick'; i++) await new Promise(resolve => setImmediate(resolve));
  assert.equal(p.$('start').disabled, false);
  for (let i = 0; i < 40 && !ready(); i++) await new Promise(resolve => setImmediate(resolve));
  assert.equal(ready(), true);
  assert.deepEqual(plain(state.region), { kind: 'box', box: [10, 20, 310, 320] }, 'filled in once Start goes ahead');
  await asked; await settle(30);
  assert.deepEqual(plain(p.calls.controls), [], 'leaving is owned by the engine');
  assert.equal(state.mode, 'pick'); assert.equal(state.sample.id, 's');
  assert.deepEqual(plain(state.region), { kind: 'box', box: [10, 20, 310, 320] }, 'the same region');
  assert.deepEqual(plain(state.prior), { cells: { kind: 'column', column: 'other' }, nuclei: null }, 'the same starting cells');
  assert.equal(p.$('prior-choice').value, '1');
  assert.equal(state.schedule, undefined, 'no schedule to choose');
  assert.deepEqual(plain(p.calls.optionsSet), { 'voxel-layers': 3, 'cell-compactness': 0.02 }, 'its options plus the suggested setting');
  assert.deepEqual(plain(p.calls.explicit), ['cell-compactness'], 'the suggested setting is named even at its default');
  assert.equal(p.calls.optionsSaved, false, 'not remembered as the picker\'s options until Start');
  assert.equal(p.$('notice').textContent, copy('newRunReady', { setting: 'tighter shapes' }));
  p.$('start').onclick(); await settle(30);
  assert.equal(p.calls.created?.length, 4, '(sample, region, prior, options)');
  assert.equal(p.calls.optionsSaved, true);

});

test('the mixing check: a click answers only its own run; a failed trial puts the setting back; a pause that fails sets up nothing', async () => {
  // Undo, then another run opened before the trial's end was sent: the other run's trial is never ended for it.
  const p = page();
  await settle(20);
  const { qcClient } = p.main();
  const panel = vm.runInContext('qcPanel', p.context);
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 1, iter: 1, summary: qcSummary(1, { suggestion: r4Card }) });
  await settle(20);
  await panel.options.onTry(r4Card); await settle(20);
  assert.equal(qcClient.trial.id, 1);
  vm.runInContext('knobs.beforeUndo = () => { state.run = { ...state.run, id: "r2" }; }', p.context);
  await panel.options.onUndo(qcClient.trial); await settle(20);
  assert.deepEqual(plain(p.calls.undos.at(-1)), ['cell-compactness', 0.04]);
  assert.ok(!p.qc.calls.some(c => c[0] === 'end'), 'no end sent once another run is open');
  // The trial cannot be registered (409): the knob goes back as it was, and the notice says so.
  const q = page({ qc: qcEngine({}, { failTrial: true }) });
  await settle(20);
  const card = vm.runInContext('qcPanel', q.context);
  await card.options.onTry(r4Card); await settle(20);
  assert.deepEqual(plain(q.calls.tries), [['cell-compactness', 0.02, 0.04]]);
  assert.deepEqual(plain(q.calls.undos), [['cell-compactness', 0.04]], 'put back: nothing stays changed without a readout to undo it');
  assert.equal(q.$('notice').textContent, QC_TEXT.trialFailedUndone);
  const r = page(); await settle(20);
  const asked = vm.runInContext('qcPanel', r.context).options.onNewRun(r4Card); await settle();
  await r.$('leave-stay').click(); await asked;
  assert.deepEqual(plain(r.calls.controls), []); assert.equal(r.main().state.mode, 'run');
  assert.equal(r.calls.optionsSet, null);

});

test('the mixing check: New run carries the values set during the run (not the temperature) and says when its starting cells are not offered', async () => {
  const p = page();
  await settle(20);
  const { state } = p.main();
  Object.assign(state.run, { region: { kind: 'box', box: [10, 20, 310, 320], bbox: [10, 20, 310, 320] }, prior: { cells: { kind: 'column', column: 'gone' }, nuclei: null },
    options: { 'cell-compactness': 0.08 }, live_values: { 'prior-weight': 2, temperature: 0.3 } });
  state.health = { ...state.health, proseg: true };
  state.runs = [{ ...state.run }];
  p.calls.imageGate = deferred();                                 // the overview image takes its time
  const changing = vm.runInContext('qcPanel', p.context).options.onNewRun({ ...r4Card, id: 'R2:cell-compactness:0.04', rule: 'R2', from: 0.08, to: 0.04 });
  await discardRun(p); await changing;
  await settle(30);
  assert.equal(vm.runInContext('qcActing', p.context), false, 'the click is done once the picker is filled in, not when its image has loaded');
  assert.deepEqual(plain(p.calls.optionsSet), { 'cell-compactness': 0.04, 'prior-weight': 2 }, 'the run as it is now, plus the setting; the schedule keeps its temperature');
  assert.equal(p.$('prior-choice').value, '-1');
  assert.equal(p.$('notice').textContent, QC_TEXT.newRunPriorMissing, 'the starting cells are not offered: said, not claimed');
  // A run that ended: New run from the drift card (any setting card); a switch by its plain change ("with drift on").
  const q = page();
  await settle(20);
  const ended = q.main().state;
  Object.assign(ended.run, { state: 'done', region: { kind: 'box', box: [10, 20, 310, 320], bbox: [10, 20, 310, 320] },
    prior: { cells: { kind: 'column', column: 'cell_id' }, nuclei: null }, options: { 'no-diffusion': true } });
  ended.health = { ...ended.health, proseg: true };
  ended.runs = [];
  const switching = vm.runInContext('qcPanel', q.context).options.onNewRun({ id: 'R3:no-diffusion:False', rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false });
  await discardRun(q); await switching;
  await settle(30);
  assert.deepEqual(plain(q.calls.controls), [], 'nothing to pause in a run that ended');
  assert.deepEqual(plain(q.calls.optionsSet), { 'no-diffusion': false });
  assert.equal(q.$('notice').textContent, copy('newRunReady', { setting: 'drift on' }));
  assert.ok(!q.$('notice').textContent.includes('with with'));
});

test('Go deeper: the mixing check\'s button and key D for the selected cell; its pushes, the check\'s paint batch, a reconnect and a new run reach the window', async () => {
  const p = page();
  await settle(20);
  const deep = p.calls.deeper;
  assert.ok(deep, 'main.js builds the window'); assert.ok(p.calls.arrows, 'and its arrows');
  assert.equal(deep.host, p.$('stage').parentElement, 'inside the stage, never over the UMAP pane or the side column');
  p.select(6);
  await settle(20);
  assert.ok(deep.selects.includes(7), 'the window follows the selection');
  // "Go deeper ›" is in the mixing check's row for the cell, bound to the window: never in the cell card.
  const button = vm.runInContext('qcPanel', p.context).deeperButton;
  assert.equal(p.calls.cells.at(-1).label, 7); assert.equal(p.calls.cells.at(-1).name, 'c6');
  assert.ok(button, 'the mixing check has "Go deeper ›"'); assert.equal(button.textContent, 'Go deeper ›'); assert.equal(button.dataset.action, 'go-deeper');
  assert.ok(!p.$('cell-card').all().some(n => n.dataset?.action === 'go-deeper' || n.className?.includes('qc-card-line')), 'nor its line');
  assert.ok(deep.bound.includes(7));
  button.click(); assert.deepEqual(deep.toggles, [7]);
  // The window opened or closed: its button says so (bound again), with "This iteration" collapsed too.
  p.$('now-panel').open = false;
  const bound = deep.bound.length; deep.options.onChange(); assert.equal(deep.bound.length, bound + 1);
  // Key D: open or close for the selected cell (also while the mixing check and "This iteration" are collapsed).
  p.$('qc-panel').open = false;
  for (const fn of p.context.document.listeners.get('keydown')) fn({ key: 'd', target: p.$('stage'), preventDefault() {} });
  assert.deepEqual(deep.toggles, [7, 7]);
  p.$('now-panel').open = true; p.$('qc-panel').open = true;
  // Its WebSocket event goes to the window, not to the check's client.
  const result = { ev: 'deeper', run: 'r1', seq: 1, iter: 1, cell: 6, model: 'm1', result: { v: 1, state: 'empty' } };
  p.socket().message(result); assert.deepEqual(deep.events.at(-1), result);
  // The check's paint batch tells the window the frame or the check changed.
  const changes = deep.changes;
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  assert.ok(deep.changes > changes, 'renderQc hands every snapshot to the window (its own throttle applies)');
  // A source neighbour under the pointer: lifted in the window, and its line in the stage tooltip (checked as the check's copy).
  deep.sourceLabel = 7;
  p.$('stage').emit('pointermove', { offsetX: 100, offsetY: 30, pointerId: 1, button: 0 });
  assert.equal(deep.hovers.at(-1), 7);
  const lines = p.$('tooltip').children.map(n => n.textContent);
  const line = p.$('tooltip').children.find(n => /seems to come from here/.test(n.textContent));
  assert.ok(line, lines.join(' | ')); assert.ok('qcCopy' in line.dataset, 'the harness checks it for banned words');
  // The harness's view of the page carries the window's snapshot, and qcTexts its words and the arrows' labels.
  const texts = vm.runInContext('window.sopaCompanion.qcTexts()', p.context);
  assert.ok(texts.includes('Go deeper words') && texts.includes('46 %'));
  // A reconnect asks again; another run closes the window.
  p.socket().connection(true); assert.ok(deep.refreshed >= 1);
  const resets = deep.resets.length;
  vm.runInContext('resetQc(null)', p.context); assert.equal(deep.resets.length, resets + 1); assert.equal(deep.resets.at(-1), null);
  assert.equal(p.calls.cells.at(-1), null, 'the mixing check\'s row for the old run\'s cell goes at once');
});

test('the window\'s glow, focus and proseg line reach the stage; the test corner shows the panel\'s test and its presses, views and box come back', async () => {
  const p = page({ hook: true });
  await settle(20);
  const { stage, state } = p.main(), deep = p.calls.deeper, corner = p.calls.corner, panel = vm.runInContext('qcPanel', p.context);
  const staged = [];
  Object.assign(stage, { setGlow: g => staged.push(['glow', g]), setSpill: s => staged.push(['spill', s]), setFocus: f => staged.push(['focus', f]),
    replan: () => staged.push(['replan']), reducedMotion: () => true, glow: { on: true, count: 2, fps: 12 } });
  // The window's suspects glow on the stage and go with its dots; its focus keeps the cells it lights and how far the
  // rest greys (B).
  const glow = { key: 'k', points: Float32Array.of(1, 2, 0.9), giver: [5, 5] };
  deep.options.onGlow(glow); deep.options.onGlow(undefined);
  deep.options.onSpill({ key: 's', cell: 7, lit: [7, 3], points: new Float32Array(4), colors: [] }); deep.options.onSpill(null);
  assert.deepEqual(staged.splice(0).map(([k, v]) => [k, v && (v.key ?? v)]), [['glow', 'k'], ['glow', null], ['spill', 's'], ['spill', null], ['glow', null]]);
  deep.options.onFocus({ cell: 7, sources: [{ label: 3, share: 0.4 }, { label: 3, share: 0.4 }], same: [5], hot: 0, lit: [7, 3, 5], dim: 1, motion: true });
  assert.deepEqual(JSON.parse(JSON.stringify(staged.splice(0))), [['focus', { cell: 7, sources: [[3, 0.4], [3, 0.4]], same: [5], hot: 0, lit: [7, 3, 5], dim: 1, motion: true }]]);
  // Proseg's own line: the result's frame's statistics with its counted file (assigned, proseg's averaged count; the
  // counted case: below), repainted when they arrive; none without that file (one draw is not the averaged count the
  // tooltip names), nor for a cell the frame does not have.
  assert.equal(deep.options.context().cellStats(7, 1), null); assert.equal(deep.options.context().cellStats(99, 1), null);
  // How the run started (the engine's start_kind): only a start from cells alone has a start line; null until it says.
  assert.equal(deep.options.context().startKind, null);
  vm.runInContext("qcClient.startKind = 'paired'", p.context); assert.equal(deep.options.context().startKind, 'paired');
  const stats = deep.statsChanges ?? 0;
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  assert.equal(state.current, 2); assert.ok(deep.statsChanges > stats, 'the window hears of the new statistics');
  // The test corner: built once inside the stage; the panel's model goes to it, its presses to the panel, its view from
  // the engine (GET .../qc/test/view), its box to the boxes the Mixing legend keeps clear of; calm motion holds its glow still.
  assert.equal(corner.host, p.$('stage').parentElement);
  panel.options.onTestCorner({ open: true, phase: 'form' }); assert.deepEqual(corner.models, [{ open: true, phase: 'form' }]);
  const pressed = []; panel.testAction = (action, payload) => { pressed.push([action, payload]); return Promise.resolve(); };
  corner.options.onAction('setting', 'prior-weight'); corner.options.onAction('run');
  assert.deepEqual(pressed, [['setting', 'prior-weight'], ['run', undefined]]);
  const views = []; p.context.api.qcTestView = async id => { views.push(id); return { state: 'running', runs: [] }; };
  assert.deepEqual(await corner.options.fetchView('r1'), { state: 'running', runs: [] }); assert.deepEqual(views, ['r1']);
  corner.options.onBox({ left: 12, top: 372, bottom: 12, w: 494, h: 416 });
  assert.deepEqual({ ...vm.runInContext('qcBoxes', p.context).get(corner.root) }, { left: 12, bottom: 12, w: 494, h: 416 });
  corner.options.onBox(null); assert.equal(vm.runInContext('qcBoxes', p.context).has(corner.root), false);
  assert.deepEqual({ ...corner.options.stage() }, { width: 1000, height: 800 }); assert.equal(corner.options.reducedMotion(), true);
  const layouts = corner.layouts; vm.runInContext('drawOverlay()', p.context); assert.equal(corner.layouts, layouts + 1, 'kept inside the stage');
  vm.runInContext('setCalmMotion(true)', p.context);
  assert.equal(corner.options.calmMotion(), true); assert.equal(corner.refreshes, 1); assert.deepEqual(staged.splice(0), [['replan']]);
  // A start carries {cell (label - 1), flag, value} only: the engine fixes the suspects and their neighbours itself. The
  // pool's preview asks for the cell as label - 1 too, with how long the engine may wait for it.
  assert.equal(panel.options.suspects, undefined, 'the window\'s suspects are not sent');
  const posts = []; p.context.api.qcTestStart = async (id, body) => { posts.push([id, body]); return { id: 't1', state: 'preparing' }; };
  await panel.options.onTestStart(7, 'cell-compactness', 0.02);
  await panel.options.onTestStart(4, 'diffusion-probability', 0.3);
  assert.deepEqual(JSON.parse(JSON.stringify(posts)), [['r1', { cell: 6, flag: 'cell-compactness', value: 0.02 }], ['r1', { cell: 3, flag: 'diffusion-probability', value: 0.3 }]]);
  const pools = []; p.context.api.qcTestPool = async (id, cell, options) => { pools.push([id, cell, options.wait]); return { v: 1, state: 'ready' }; };
  await panel.options.onTestPreview(7, { wait: 5 }); await panel.options.onTestPreview(9);
  assert.deepEqual(pools, [['r1', 6, 5], ['r1', 8, 0]]);
  // Go deeper's "Test a setting here…": the panel's form for the window's cell, and the panel shown.
  const forms = []; panel.openTestForm = cell => forms.push(cell);
  deep.options.onTest(7, 'c6'); deep.options.onTest(9);
  assert.deepEqual(JSON.parse(JSON.stringify(forms)), [{ label: 7, name: 'c6' }, { label: 9, name: vm.runInContext('cellName(9)', p.context) }]);
  // What Go deeper reads: whether the panel can test its cell (the fake panel here has no testable(): never), and the
  // start's hold of points (PriorLayers.startHold; none here).
  assert.equal(deep.options.context().testable, false); assert.equal(deep.options.context().startHold([], 7), null);
  panel.testable = label => label === 7;
  vm.runInContext('state.selected = 7', p.context); assert.equal(deep.options.context().testable, true);
  vm.runInContext('state.selected = 0', p.context);
  assert.equal(deep.options.context().testable, true, 'nothing selected: the window\'s own cell (its focus, cell 7 above)');
  deep.options.onFocus(null); assert.equal(deep.options.context().testable, false, 'no cell at all');
  const holds = []; vm.runInContext('priors', p.context).startHold = (...args) => { holds.push(args); return Uint8Array.of(1, 2); };
  assert.deepEqual([...deep.options.context().startHold([{ x: 1, y: 2, z: 3 }], 7)], [1, 2]);
  assert.deepEqual(holds.map(([points, label]) => [points.length, label]), [[1, 7]]);
  // The harness: the corner itself, its words with the check's, and the snapshot's corner and stage glow.
  assert.equal(vm.runInContext('window.sopaCompanion.testCorner', p.context), corner);
  assert.ok(vm.runInContext('window.sopaCompanion.qcTexts()', p.context).includes('Suspects inside: 2 of 3'));
  const snap = p.snapshot();
  assert.deepEqual(snap.glow, { on: true, count: 2, fps: 12 }); assert.deepEqual(snap.qc.testCorner, { open: true, phase: 'form' });
});

test('the starting outline, the corner\'s box for the window\'s camera, the test square, the legend beside the corner, the corner\'s colours, one setting a test', async () => {
  const p = page({ hook: true });
  await settle(20);
  const { stage, analysis, priors } = p.main(), deep = p.calls.deeper, corner = p.calls.corner, panel = vm.runInContext('qcPanel', p.context);
  // The window's Show asks for its cell's starting outline (the prior's by the cell's name, else its starting labels
  // traced: prior-layers.js); until the prior holds it the stage has none and the overlay's next view asks again; Show
  // off, the dots going or another run clear it.
  const sent = p.calls.outlines.length;   // (a run opening clears it: resetQc)
  deep.options.onStartOutline({ cell: 7, name: 'c6' });
  assert.deepEqual(p.calls.startAsks, [['c6', 7]]); assert.deepEqual(p.calls.outlines.slice(sent), [null]); assert.equal(stage.startOutline, null);
  priors.starts.set('c6', { points: Float32Array.of(0, 0, 4, 0, 4, 4, 0, 4), closed: true, source: 'prior' });
  vm.runInContext('drawOverlay()', p.context);
  assert.deepEqual(p.calls.startAsks.at(-1), ['c6', 7]); assert.equal(stage.startOutline.count, 4);
  const asks = p.calls.startAsks.length; vm.runInContext('drawOverlay()', p.context);
  assert.equal(p.calls.startAsks.length, asks, 'drawn: not asked again');
  deep.options.onStartOutline(null); assert.equal(stage.startOutline, null);
  deep.options.onStartOutline({ cell: 7, name: 'c6' }); assert.equal(stage.startOutline.count, 4);
  deep.options.onSpill(null); assert.equal(stage.startOutline, null, 'the dots off: the outline goes with them');
  deep.options.onStartOutline({ cell: 7 }); assert.deepEqual(p.calls.startAsks.at(-1), ['c6', 7], 'no name from the window: the run\'s');
  vm.runInContext('resetQc(null)', p.context); assert.equal(stage.startOutline, null, 'another run: none');
  assert.deepEqual(p.snapshot().startOutline, null);
  deep.options.onStartOutline({ cell: 7, name: 'c6' }); assert.deepEqual(plain(p.snapshot().startOutline), { count: 4, closed: true });
  // The window's camera move keeps clear of the open test corner: the corner's own box (null while closed).
  assert.equal(deep.options.context().cornerBox(), null);
  corner.place = { left: 12, top: 372, bottom: 12, w: 494, h: 416 };
  assert.equal(deep.options.context().cornerBox(), corner.place);
  // The Mixing legend out of the open corner's way (after 5b's first browser run: moved above the corner, it met the
  // Mixing chip at the top left): just right of the corner at the stage's bottom (main.js legendSpot), else hidden while
  // the corner is open; never over the HUD or the chip; back in its place once the corner closes or leaves it free. At
  // once: no motion.
  const legendRoot = vm.runInContext('mixingLegend.root', p.context), legend = vm.runInContext('qcBoxes.get(mixingLegend.root)', p.context);
  const place = () => [legendRoot.style.left, legendRoot.style.bottom, legendRoot.style.visibility];
  Object.assign(legend, { w: 199, h: 40 });   // as its ResizeObserver reports it
  corner.options.onBox(corner.place);         // 494 x 416 at the bottom left of the 1000 x 800 stage
  assert.deepEqual(place(), ['514px', '12px', '']); assert.deepEqual([legend.left, legend.bottom, legend.off], [514, 12, false]);
  // The answer beside the views (783 wide): a key 290 x 159 has no room right of it on this stage: hidden; a key 170
  // wide has (803 + 170 <= 1000 - 12).
  Object.assign(legend, { w: 290, h: 159 });
  corner.options.onBox({ left: 12, top: 364, bottom: 12, w: 783, h: 424 });
  assert.deepEqual(place(), ['12px', '12px', 'hidden']); assert.equal(legend.off, true);
  assert.equal(legendRoot.hidden, false, 'its own hidden stays MixingLegend.update\'s');
  Object.assign(legend, { w: 170, h: 40 }); vm.runInContext('placeLegend()', p.context);
  assert.deepEqual(place(), ['803px', '12px', '']);
  Object.assign(legend, { w: 186, h: 40 }); vm.runInContext('placeLegend()', p.context);
  assert.deepEqual(place(), ['12px', '12px', 'hidden'], 'one past the stage\'s margin: hidden, never cut');
  // The corner moved up off the legend's place: the legend back in its own.
  corner.options.onBox({ left: 12, top: 100, bottom: 384, w: 494, h: 316 });
  assert.deepEqual(place(), ['12px', '12px', '']); assert.equal(legend.off, false);
  // Never over the HUD or the chip: on a short stage the spot right of the corner's strip meets a wide chip: hidden; a
  // narrower chip leaves it free.
  const hud = vm.runInContext("qcBoxes.get($('hud'))", p.context), chip = vm.runInContext("qcBoxes.get($('qc-chip'))", p.context);
  p.$('hud').hidden = false; p.$('qc-chip').hidden = false;
  Object.assign(hud, { w: 330, h: 100 }); Object.assign(chip, { w: 700, h: 30 });   // the chip: 12-712 x 118-148
  stage.cssHeight = 300; Object.assign(legend, { w: 290, h: 159 });
  corner.options.onBox({ left: 12, top: 233, bottom: 12, w: 320, h: 55 });   // the strip; the legend's spot: 340-630 x 129-288
  assert.deepEqual(place(), ['12px', '12px', 'hidden']);
  chip.w = 300; vm.runInContext('placeLegend()', p.context);
  assert.deepEqual(place(), ['340px', '12px', '']);
  const spot = (...args) => JSON.parse(JSON.stringify(vm.runInContext('legendSpot', p.context)(...args)));
  assert.deepEqual(spot(null, { w: 290, h: 159 }, { width: 1000, height: 800 }), { left: 12, bottom: 12 }, 'no corner: its own place');
  assert.equal(spot({ left: 12, top: 300, bottom: 12, w: 494, h: 488 }, { w: 290, h: 159 }, { width: 700, height: 800 }), null, 'no room: hidden');
  assert.deepEqual(spot({ left: 12, bottom: 12, w: 494, h: 488 }, null, { width: 1000, height: 800 }), { left: 514, bottom: 12 }, 'its most size without one; the top from the bottom');
  stage.cssHeight = 800;
  corner.options.onBox(null); assert.deepEqual(place(), ['12px', '12px', '']); assert.deepEqual([legend.left, legend.bottom, legend.off], [12, 12, false]);
  assert.doesNotMatch(readFileSync(new URL('../src/main.js', import.meta.url), 'utf8'), /mixingLegend\.dock\(/, 'never above the corner any more');
  // The test's square on the stage while the test exists (the panel's model: test.square), gone with the test.
  panel.options.onTestCorner({ open: true, phase: 'form', test: null });
  assert.equal(p.calls.squares.at(-1), null);
  panel.options.onTestCorner({ open: true, phase: 'running', test: { square: [100, 200, 520, 620] } });
  assert.deepEqual(plain(p.calls.squares.at(-1)), { box: [100, 200, 520, 620] }); assert.deepEqual(plain(p.snapshot().qc.testSquare), [100, 200, 520, 620]);
  panel.options.onTestCorner({ open: false });
  assert.equal(p.calls.squares.at(-1), null); assert.equal(p.snapshot().qc.testSquare, null);
  // The corner's cells in the stage's colours: a cell of your run in its identity colour, under Colour by: Cluster
  // its cluster's (the stage's grey without one); Colour by and each new analysis draw the corner again.
  assert.equal(corner.options.colorOf(7), vm.runInContext('swatch(7)', p.context));
  analysis.applied = { clusters: Int16Array.of(-1, 0, 0, 1, 1, 0, 2), summary: { clusters: [{ id: 0, color: 3 }, { id: 1 }, { id: 2, color: 5 }] } };
  const recolors = corner.recolors;
  vm.runInContext("colourBy('cluster')", p.context); assert.equal(corner.recolors, recolors + 1);
  assert.deepEqual([7, 2, 4, 1].map(label => corner.options.colorOf(label)), [clusterColorCSS(5), clusterColorCSS(3), clusterColorCSS(1), '#66717e']);
  analysis.onReadoutChange(); assert.equal(corner.recolors, recolors + 2);
  // One setting a test: whatever else the panel were to pass, the start carries {cell, flag, value} only.
  const posts = []; p.context.api.qcTestStart = async (id, body) => { posts.push(body); return { id: 't2', state: 'preparing' }; };
  await panel.options.onTestStart(7, 'prior-weight', 0.072, [], [{ flag: 'prior-weight', value: 0.072 }, { flag: 'cell-compactness', value: 0.16 }]);
  assert.deepEqual(JSON.parse(JSON.stringify(posts)), [{ cell: 6, flag: 'prior-weight', value: 0.072 }]);
  // New run with this: the picker filled in as the run had it, with the tested setting at its value from the start (the
  // suggestion's New run); Keep and Cancel through qcTestEnd, never "use".
  assert.equal(typeof panel.options.onTestNewRun, 'function'); assert.equal(typeof panel.options.onTestShow, 'function');
  assert.equal(panel.options.onTestAnswer.length, 1, 'Keep or Close: the test alone');
});

test('Compare explains with the check off, opens over the run page on the frame on screen; underneath nothing draws or asks until Back, then it catches up', async () => {
  const off = page({ qc: qcEngine({ enabled: false }) });
  await settle(20);
  assert.equal(off.$('compare-button').hidden, false, 'a Proseg Live run: the button is there, never greyed out');
  assert.equal(off.$('compare-button').textContent, COMPARE_TEXT.button); assert.equal(off.$('compare-button').title, COMPARE_TEXT.buttonTooltip);
  await off.$('compare-button').click();
  assert.deepEqual(off.calls.compare.opens, [], 'the check off: nothing opens, nothing is asked');
  const note = off.$('compare-button').nextSibling;
  assert.equal(note.hidden, false); assert.match(note.textContent, new RegExp(COMPARE_TEXT.needsCheck.replace(/[.:]/g, '\\$&')));
  await vm.runInContext('qcClient.setEnabled(true)', off.context); await settle(10);
  assert.equal(note.hidden, true, 'switched on: the note goes');

  const p = page({ hook: true });
  await settle(20);
  const { state, stage, timeline, priors, paint, analysis } = p.main(), view = p.calls.compare, corner = p.calls.corner, deeper = p.calls.deeper;
  // Layers > Load Other Mask: the run's sample (a SpatialData one) to the store, the group in the picker's card too.
  assert.deepEqual(plain(priors.segments.samples), ['s']); assert.deepEqual(p.calls.segHosts, ['picker-segmentations']);
  assert.deepEqual(plain(p.snapshot().segmentations), { sample: 's', note: null, loaded: [] });
  // A save into a sample that ends (ZarrSave's onSaved): the store reads that sample's list anew.
  p.calls.zarrSave.onSaved({ id: 'r1', sample: 's' }); assert.deepEqual(plain(priors.segments.saved), ['s']);
  assert.equal(view.host, p.$('compare-host'));
  assert.equal(state.current, 1);
  await p.$('compare-button').click();
  assert.deepEqual(plain(view.opens), [{ run: 'r1', seq: 1, iteration: 1, sampleId: 's', box: [0, 0, 4, 4], kept: null }], 'the frame on screen');
  assert.deepEqual([stage.covered, timeline.covered, corner.covered, p.calls.panelCovered], [true, true, true, true]);
  assert.deepEqual([analysis.covered, deeper.covered], [true, true], 'the UMAP pane and Go deeper\'s window hold still too');
  assert.deepEqual(plain(p.snapshot().compare), { open: true, state: 'working', columns: [], area: null });
  assert.ok(vm.runInContext('window.sopaCompanion.qcTexts()', p.context).includes('13 in 100 cells look mixed'));
  assert.equal(vm.runInContext('window.sopaCompanion.compare', p.context), view);
  // Covered: new frames are kept but none is shown (no frame fetched, no panel written), the check renders nothing.
  const shown = p.calls.setFrame.length, panels = p.calls.panel.length, huds = p.$('hud-iter').textContent;
  for (let seq = 2; seq < 5; seq++) { p.socket().message({ ev: 'frames', run: 'r1', from: seq, frames: [p.frame(seq)] }); await settle(10); }
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 1, summary: qcSummary(1) }); await settle(20);
  vm.runInContext('qcClient.changed(true)', p.context);   // a result rendered at once (a paused run's): it waits too
  assert.deepEqual([state.frames.length, state.current, p.calls.setFrame.length, p.$('hud-iter').textContent], [5, 1, shown, huds]);
  assert.equal(p.calls.panel.length, panels, 'the mixing check renders nothing under it');
  // The panels' batch asks for no animation frame meanwhile: a run event's job waits for the view to close.
  p.socket().message({ ev: 'run', run: { ...state.run } }); await settle(10);
  assert.deepEqual([paint.covered, paint.raf, paint.jobs.has('run')], [true, null, true]);
  // A notice raised meanwhile shows inside the view (the top layer), not under it.
  vm.runInContext('notice("Connection failed. Please try again.")', p.context);
  assert.equal(p.$('notice').parentElement, view.root);
  // Back: everything again, the newest frame shown (live), the check rendered; calm motion reaches the view.
  view.close(); await settle(20);
  assert.deepEqual([stage.covered, timeline.covered, corner.covered, p.calls.panelCovered, paint.covered], [false, false, false, false, false]);
  assert.deepEqual([analysis.covers.at(-1), deeper.covers.at(-1)], [[false, true], [false, true]], 'uncovered with a catch-up: what they held is shown');
  assert.equal(paint.jobs.has('run'), false, 'what waited has run');
  assert.equal(state.current, 4); assert.equal(p.calls.setFrame.at(-1), 4); assert.ok(p.calls.panel.length > panels);
  vm.runInContext('notice("Saved.")', p.context); assert.notEqual(p.$('notice').parentElement, view.root, 'back on the page');
  vm.runInContext('setCalmMotion(true)', p.context); assert.equal(view.refreshes, 1);
  // Paused (not following): what the check rendered meanwhile comes when the view closes; the frame stays.
  vm.runInContext('setLive(false)', p.context);
  await p.$('compare-button').click();
  const before = p.calls.panel.length;
  vm.runInContext('qcClient.changed(true)', p.context); assert.equal(p.calls.panel.length, before);
  view.close(); await settle(5);
  assert.ok(p.calls.panel.length > before, 'rendered once uncovered'); assert.equal(state.current, 4);
  // Another run opened over it (openRun), or the run gone (enterPick): the view closes with it, no catch-up for the old.
  await p.$('compare-button').click(); assert.equal(view.isOpen, true);
  vm.runInContext('openRun("r1")', p.context); await settle(20);
  assert.deepEqual([view.isOpen, stage.covered, state.mode], [false, false, 'run']);
  assert.deepEqual([analysis.covers.at(-1), deeper.covers.at(-1)], [[false, false], [false, false]], 'no catch-up for the run left');
  await p.$('compare-button').click(); assert.equal(view.isOpen, true);
  const sample = p.context.api.sample; p.context.api.sample = async id => ({ ...(await sample(id)), kind: 'zarr' });
  p.socket().message({ ev: 'gone', run: 'r1' }); await settle(20);
  assert.deepEqual([view.isOpen, stage.covered, state.mode], [false, false, 'pick']);
  assert.deepEqual(plain(priors.segments.samples).slice(-2), [null, 's'], 'the picker hands the store its sample anew');
  // Without a run, a press says to open one.
  await p.$('compare-button').click();
  assert.equal(view.opens.length, 4); assert.match(p.$('compare-button').nextSibling.textContent, /Open a run first/);
});

test('the mixing check: "Look at its cells" (a card whose cause is not known) moves the view to its area (nothing drawn), then its most mixed cell with Go deeper open on it', async () => {
  const p = page();
  await settle(20);
  const { state, qcClient, cellStats, stage } = p.main();
  const fits = []; stage.fit = (rect, animate) => fits.push([rect, animate]);
  const outline = [[0, 0], [10, 0], [10, 10], [0, 10]];
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2, { areas: [{ L: 'A', tag: 'already', per100: 23, outline }] }) });
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  assert.equal(state.current, 2); assert.equal(qcClient.applied.seq, 2);
  // The frame's cell centres (µm): cells 1, 4, 5 and 7 inside the area, 2, 3 and 6 outside. The result's values are
  // 10, 80, 255 (not judged), 30, 0, 90, 76: the most mixed judged cell inside is 7, not 6 (outside) or 3 (not judged).
  cellStats.current = { ...cellStats.current, seq: 2, x: Float32Array.of(2, 20, 5, 8, 1, 30, 9, 50), y: Float32Array.of(2, 5, 30, 8, 9, 3, 1, 50) };
  const deep = p.calls.deeper;
  vm.runInContext("qcInspect({ rule: 'R6' }, { L: 'A', outline: [[0, 0], [10, 0], [10, 10], [0, 10]] })", p.context);
  assert.deepEqual(plain(fits), [[plain(flyTarget(outline)), true]], 'the view moves there at once (at least 300 µm across)');
  await settle(20);
  // Live, the values land with the next snapshot's paint (at most a second later): that paint, now.
  qcClient.frame(); await settle(20);
  await new Promise(resolve => setTimeout(resolve, 5));   // the cell is picked on the next tick, never inside a render
  assert.ok(p.qc.calls.some(c => Array.isArray(c) && c[0] === 'cells' && c[1] === 2), 'the result shown\'s own values are asked for');
  assert.deepEqual(deep.opens, [7], 'Go deeper opens on the area\'s most mixed judged cell');
  assert.equal(state.selected, 7, 'and the selection moves there');
  // Without that frame's cell centres nothing is picked (only the view moves there).
  vm.runInContext('qcInspecting = null', p.context);
  cellStats.current = { ...cellStats.current, seq: 1 };
  vm.runInContext("qcInspect({ rule: 'R1' }, { L: 'A', outline: [[0, 0], [10, 0], [10, 10], [0, 10]] })", p.context);
  await settle(20);
  assert.deepEqual(deep.opens, [7]);
  // R1's "Show the outlines of Xenium's cells here": the starting outlines on, and the view at the card's area (its
  // outline the result shown's, by its letter); an area the result does not have leaves the view where it is.
  fits.length = 0;
  vm.runInContext("qcPanel.options.onShowCells(true, { start: true, area: { L: 'A' } })", p.context);
  assert.equal(vm.runInContext('priors.settings.xenium.on', p.context), true);
  assert.deepEqual(plain(fits), [[plain(flyTarget(outline)), true]]);
  vm.runInContext("qcPanel.options.onShowCells(true, { start: true, area: { L: 'Z' } })", p.context);
  assert.equal(fits.length, 1, 'no such area: the view stays');
});

test('the mixing check: while the shown result\'s values load, the cell\'s row is told they are on the way, and no line is made up', async () => {
  const qc = qcEngine();
  const gate = deferred();
  qc.api.qcCells = async (id, seq) => { qc.calls.push(['cells', seq]); await gate.promise; return Uint8Array.of(10, 80, 255, 30, 0, 90, 76); };
  const p = page({ qc });
  await settle(20);
  const { state, qcClient } = p.main();
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2) });
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  assert.equal(qcClient.applied.seq, 2);
  p.select(6); await settle(20);
  vm.runInContext('updateCellCard()', p.context); await settle(5);
  // The mixing check's row (qc-panel.js setCell keeps a line it has in its place, invisible, while they load).
  const row = () => p.calls.cells.at(-1);
  assert.equal(row().label, 7);
  assert.equal(row().line, '', 'nothing is claimed before the result\'s own values are in');
  assert.equal(row().loading, true); assert.ok(!('standIn' in row()), 'no stand-in line is made up');
  gate.resolve(); await settle(20);
  qcClient.frame(); await settle(20);
  vm.runInContext('updateCellCard()', p.context); await settle(5);
  assert.equal(row().loading, false, 'shown once they land');
  assert.equal(row().line, cardLine(76, 76, 'xenium'));
  assert.match(row().line, /^Mixing: the model reads about 30 % of its RNA as coming from neighbouring cells/);
  void state;
});

test('the mixing check\'s row for the selected cell: in its panel with the check on and off, built once, never in the cell card', async () => {
  const p = page({ realPanel: true });
  await settle(20);
  const { qcClient } = p.main(), panel = p.$('qc-panel'), card = p.$('cell-card');
  const row = panel.all().find(n => n.className === 'qc-cell'), body = panel.children.find(n => n.className === 'qc-body');
  const deep = () => panel.all().find(n => n.dataset?.action === 'go-deeper'), line = row.children.find(n => n.className?.includes('qc-card-line'));
  assert.equal(body.children[0], row, 'first in the panel\'s body: nothing above it changes while it shows');
  assert.equal(row.hidden, true, 'no cell selected: no row');
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2) });
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  p.select(6); await settle(20);
  vm.runInContext('updateCellCard()', p.context); await settle(20); qcClient.frame(); await settle(20);
  vm.runInContext('updateCellCard()', p.context);
  assert.equal(row.hidden, false); assert.equal(row.dataset.cell, 'c6'); assert.match(row.textContent, /^c6/);
  // The panel's headline shows, with its meaning line: the cell's line leaves that note out of its words (its tooltip).
  const headline = panel.all().find(n => n.className === 'qc-headline');
  assert.equal(headline.hidden, false);
  assert.equal(line.textContent, cardLine(76, 76, 'xenium', { note: false })); assert.equal(line.title, QC_TEXT.cardNote);
  assert.equal(line.hidden, false); assert.equal(line.style.visibility, '');
  const button = deep();
  assert.equal(button?.textContent, 'Go deeper ›'); assert.equal(button.parentElement, row); assert.ok(p.calls.deeper.bound.includes(7));
  assert.deepEqual(row.children.map(n => n.tagName), ['HEADER', 'BUTTON', 'P'], 'the button under the name, above the line');
  // The cell card: its name, readout and history only.
  assert.equal(card.hidden, false); assert.match(card.textContent, /c6/); assert.match(card.textContent, /µm² · /);
  assert.ok(!card.all().some(n => n.dataset?.action === 'go-deeper' || n.className?.includes('qc-card-line')));
  assert.doesNotMatch(card.textContent, /Mixing|Go deeper/);
  // Every frame and every result update the row in place: the same nodes, and its words only where they changed.
  const nodes = [...row.childNodes], text = line.childNodes;
  for (let seq = 3; seq < 6; seq++) {
    p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq, iter: seq, summary: qcSummary(seq) });
    p.socket().message({ ev: 'frames', run: 'r1', from: seq, frames: [p.frame(seq)] });
    await settle(20);
  }
  assert.equal(p.main().state.current, 5);
  assert.deepEqual(row.childNodes, nodes, 'never rebuilt'); assert.equal(deep(), button);
  assert.equal(line.childNodes, text, 'the same words are not written again');
  // A result from too few judged cells has no headline: the cell's line says the note itself.
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 6, iter: 6, summary: qcSummary(6, { judged: 20 }) });
  p.socket().message({ ev: 'frames', run: 'r1', from: 6, frames: [p.frame(6)] });
  await settle(20);
  assert.equal(qcClient.applied.seq, 6); assert.equal(headline.hidden, true);
  assert.equal(line.textContent, cardLine(76, 76, 'xenium')); assert.equal(line.title, '');
  assert.deepEqual(row.childNodes, nodes); assert.equal(deep(), button);
  // Switched off: the compact panel keeps the cell and its button (the window then offers the check's switch).
  await vm.runInContext('qcClient.setEnabled(false)', p.context); await settle(20);
  assert.ok(panel.classList.contains('qc-off'));
  assert.equal(row.hidden, false); assert.equal(line.hidden, true); assert.equal(deep(), button); assert.equal(button.hidden, false);
  button.click(); assert.deepEqual(p.calls.deeper.toggles, [7]);
  // Nothing selected (Esc): no row.
  for (const fn of p.context.document.listeners.get('keydown')) fn({ key: 'Escape', target: p.$('stage'), preventDefault() {} });
  assert.equal(row.hidden, true);
  // A run without a sample has no mixing check to go deeper with: the card only.
  const q = page({ realPanel: true, runOverride: { sample: null } });
  await settle(20);
  q.select(6); await settle(20);
  assert.equal(q.$('qc-panel').all().find(n => n.className === 'qc-cell').hidden, true);
  assert.equal(q.$('cell-card').hidden, false);
});

test('the mixing check at frame 0 while its sample waits or learns: the cell\'s row holds no invisible line all that time', async () => {
  // The start line's values come only once the sample is learned (minutes): an invisible stand-in for the line was an
  // empty box in the row all that while.
  for (const [status, extra] of [['waiting', { reason: 'gpu_memory' }], ['learning', { progress: { step: 3, of: 50 }, eta_s: 90 }]]) {
    const p = page({ realPanel: true, qc: qcEngine({ state: status, sample_state: status, model: null, start: null, ...extra }) });
    await settle(20);
    vm.runInContext('setLive(false); show(0)', p.context); await settle(20);
    p.select(6); await settle(20);
    const { qcClient } = p.main(), row = p.$('qc-panel').all().find(n => n.className === 'qc-cell');
    const line = row.children.find(n => n.className?.includes('qc-card-line'));
    assert.equal(qcClient.view.status, status); assert.equal(qcClient.view.atStart, true);
    assert.equal(row.hidden, false, 'the cell and its "Go deeper ›"'); assert.equal(row.dataset.cell, 'c6');
    assert.equal(line.hidden, true, `${status}: no line, nor an invisible stand-in for one`); assert.equal(line.textContent, '');
  }
  // Nor the line a cell had: shown for a result, then its sample learns again and frame 0's values do not come meanwhile.
  const qc = qcEngine(), cells = qc.api.qcCells;
  qc.api.qcCells = (id, seq) => seq === 'start' ? new Promise(() => {}) : cells(id, seq);
  const p = page({ realPanel: true, qc });
  await settle(20);
  p.socket().message({ ev: 'qc', run: 'r1', state: 'ready', seq: 2, iter: 2, summary: qcSummary(2) });
  p.socket().message({ ev: 'frames', run: 'r1', from: 2, frames: [p.frame(2)] });
  await settle(20);
  p.select(6); await settle(20); p.main().qcClient.frame(); await settle(20);
  const line = p.$('qc-panel').all().find(n => n.className?.includes('qc-card-line'));
  assert.equal(line.hidden, false); assert.match(line.textContent, /^Mixing: the model reads about 30 %/);
  p.socket().message({ ev: 'qc', sample: 's', state: 'learning', progress: { step: 3, of: 50 }, eta_s: 90 });
  vm.runInContext('setLive(false); show(0)', p.context); await settle(20);
  assert.equal(p.main().qcClient.view.status, 'learning'); assert.equal(p.main().qcClient.view.atStart, true);
  assert.equal(line.hidden, true, 'no line held, invisible, while the sample learns'); assert.equal(line.style.visibility, '');
});

test('another run opened: the mixing check\'s row for the old run\'s cell goes at once, before the new run\'s first frame', async () => {
  const p = page({ realPanel: true });
  await settle(20);
  p.select(6); await settle(20);
  const row = p.$('qc-panel').all().find(n => n.className === 'qc-cell');
  assert.equal(row.hidden, false); assert.equal(row.dataset.cell, 'c6');
  // A new run on the same sample, still preparing: no frame to show yet.
  p.context.api.run = async id => ({ ...p.main().state.run, id, state: 'preparing', frames: 0 });
  p.context.api.frames = async () => ({ from: 0, frames: [] });
  await vm.runInContext("openRun('r2')", p.context);
  assert.equal(p.main().state.run.id, 'r2'); assert.equal(p.main().state.selected, 0);
  assert.equal(p.$('qc-panel').hidden, false, 'its panel shows at once');
  assert.equal(row.hidden, true, 'without the old run\'s cell, from the start');
  await settle(20); assert.equal(row.hidden, true);
});

test('the mixing check collapsed: its row is not refreshed for every frame (nor the cell\'s values asked for), and is brought up to date when it opens', async () => {
  const p = page();
  await settle(20);
  const { qcClient, state } = p.main();
  p.select(6); await settle(20);
  assert.equal(p.calls.cells.at(-1).label, 7);
  let asked = 0; const value = qcClient.cellValue.bind(qcClient);
  qcClient.cellValue = label => { asked++; return value(label); };
  p.$('qc-panel').open = false;
  const handed = p.calls.cells.length;
  for (let seq = 2; seq < 5; seq++) { p.socket().message({ ev: 'frames', run: 'r1', from: seq, frames: [p.frame(seq)] }); await settle(20); }
  assert.equal(state.current, 4);
  assert.equal(p.calls.cells.length, handed, 'nothing handed to a collapsed panel'); assert.equal(asked, 0, 'nor its values asked for');
  p.$('qc-panel').open = true; p.$('qc-panel').emit('toggle');
  assert.equal(p.calls.cells.length, handed + 1); assert.equal(p.calls.cells.at(-1).label, 7); assert.equal(asked, 1);
  // A cleared selection empties it, collapsed or not.
  p.$('qc-panel').open = false;
  vm.runInContext('state.selected = 0; updateCellCard();', p.context);
  assert.equal(p.calls.cells.at(-1), null);
});

test('the cell\'s history landing rebuilds only the card\'s history part, and a press on "Go deeper ›" made meanwhile is never lost', async () => {
  // Rebuilding the whole card when the history arrived replaced its button between a press and its release (a real X
  // click: 30 ms apart), so the click made right after selecting a cell did nothing (desktop smoke, 2026-10-19). The
  // button is now the mixing check's (its row is never rebuilt: qc-panel.test.mjs); the card is still built once per cell.
  const p = page({ realPanel: true });
  await settle(20);
  const gate = deferred(), real = p.context.fetch;
  p.context.fetch = async (url, o) => { if (/cellhistory/.test(url)) await gate.promise; return real(url, o); };
  p.select(6); await settle(20);
  const card = p.$('cell-card'), deep = () => p.$('qc-panel').all().find(n => n.dataset?.action === 'go-deeper');
  const button = deep(), header = card.children[0];
  assert.ok(button && !button.hidden, 'the button is there while the history loads');
  assert.match(card.textContent, /Loading cell history/);
  gate.resolve(); await settle(20);
  assert.match(card.textContent, /Area \(µm²\)/, 'the history landed');
  assert.doesNotMatch(card.textContent, /Loading cell history/);
  assert.equal(deep(), button, 'the same button element: never replaced under a press');
  assert.equal(card.children[0], header, 'the card itself is not rebuilt');
  const canvases = card.all().filter(n => n.tagName === 'CANVAS'), ro = p.observers.find(o => canvases.every(c => o.nodes.has(c)));
  assert.equal(canvases.length, 2); assert.ok(ro, 'the new sparklines are observed for their size');
  button.click();
  assert.deepEqual(p.calls.deeper.toggles, [7]);
});

test('patch export in the full page controller scrolls into view before a pointer click', async t => {
  const patches={set:null,version:4,count:3,size:128,run_pixel_to_sample:[0,0,1,1],
    patches:[{id:1,box:[0,0,128,128]}],exported:null};
  const writes=[],listed=[],exporting=deferred();
  // The export dialog's suggestion (engine/patch_folders.py): a new folder named after the sample and its resolution.
  const suggestion={path:'/tmp/patches/sample-1-scale0',exists:false,parent:'/tmp/patches',entries:[],
    shortcuts:[{kind:'home',name:'me',path:'/home/me'}],new:'sample-1-scale0',holds:null,mixes:null,refused:null};
  const p=page({realPatches:true,stainFetch:async(url,options)=>{
    let result;
    if(url.endsWith('/patches'))result=patches;
    else if(url.endsWith('/patches/folders')){listed.push(url);result=suggestion;}
    else if(url.endsWith('/patches/export')){writes.push({url,...options});result=await exporting.promise;}
    else throw new Error(`Unexpected request: ${url}`);
    return {ok:true,status:200,json:async()=>result};
  },runOverride:{kind:'stain',state:'paused',paused:true,ack:{state:'paused'},info:null,frames:0,
    stain:{model:'cpsam'},progress:{done:1,total:1,revision:2,complete:true},box:[0,0,100,100]}});
  t.after(()=>p.main().stainPatches.reset());
  await settle(20);
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'done',paused:false}});
  await settle(10);
  const panel=p.main().stainPatches,button=p.$('stain-patch-export');
  assert.equal(button.disabled,false);
  let top=1400,covered=false;
  button.getBoundingClientRect=()=>({x:1600,y:top,width:120,height:32});
  button.scrollIntoView=()=>{top=600;};
  const query=p.context.document.querySelector;
  p.context.document.querySelector=selector=>selector==='#stain-patch-export'?button:query.call(p.context.document,selector);
  p.context.document.elementFromPoint=(x,y)=>!covered && y<900 && y>=top && y<top+32?button:null;
  const pointerClick=()=>{
    const r=button.getBoundingClientRect();
    return p.context.document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.click();
  };
  await pointerClick();assert.equal(writes.length,0);
  const ready=()=>stainControlSource?vm.runInContext(stainControlSource.replace('SELECTOR',JSON.stringify('#stain-patch-export')),p.context)
    :(button.scrollIntoView(),p.context.document.elementFromPoint(1660,top+16)===button);
  covered=true;assert.equal(ready(),false);
  covered=false;assert.equal(ready(),true);
  // The press asks where (the export dialog lists the suggestion), and posts nothing itself.
  await pointerClick();
  await settle();
  assert.equal(writes.length,0,p.$('notice').textContent);
  assert.deepEqual(listed,['/api/v1/runs/r1/stain/patches/folders']);
  assert.equal(p.$('patch-folder').open,true);assert.equal(p.$('patch-folder-path').value,suggestion.path);
  assert.equal(p.main().stainPatches.snapshot().folder.path,suggestion.path);
  const click=p.$('patch-folder-export').click();
  await settle();
  assert.equal(writes.length,1,p.$('notice').textContent);
  assert.equal(writes[0].url,'/api/v1/runs/r1/stain/patches/export');
  assert.equal(writes[0].method,'POST');
  assert.deepEqual(JSON.parse(writes[0].body),{version:4,folder:'/tmp/patches/sample-1-scale0'});
  // At work it is busy, never greyed out (explain.mjs); a press meanwhile says so instead of exporting again.
  assert.equal(button.getAttribute('aria-busy'),'true');assert.equal(button.disabled,false);
  exporting.resolve({...patches,version:5});await click;
  assert.equal(panel.data.version,5);assert.equal(p.$('patch-folder').open,false);
  assert.equal(p.$('notice').textContent,'');
});

test('stain runs use tile updates, plain controls and shapes saving in the full page controller', async () => {
  const p = page({runOverride:{kind:'stain',mode:'tiled',info:null,frames:0,progress:{done:1,total:4,revision:1},
    stain:{model:'cpsam_20250102'},box:[0,0,100,100],summary:{cells:7,area_um2:{median:20,p10:10,p90:30}}}});
  await settle(25);
  assert.equal(p.main().state.run.kind,'stain');
  assert.equal(p.$('notice').textContent,'');
  for(const id of ['knobs-panel','qc-panel','timeline']) assert.equal(p.$(id).hidden,true,id);
  assert.equal(p.$('run-stop').textContent,'Stop');
  assert.match(p.$('run-facts').textContent,/1 \/ 4/);
  assert.match(p.$('run-facts').textContent,/Modelcpsam_20250102/);
  assert.ok(p.calls.stain.includes(1));
  const run = {...p.main().state.run,state:'done',progress:{done:4,total:4,revision:5,resolved:true,complete:true},
    save_defaults:{shapes_key:'cellpose_boundaries_1',table_key:null},outputs:[]};
  p.socket().message({ev:'run',run}); await settle(20);
  assert.ok(p.$('run-toggle').disabled);
  assert.match(p.$('results').textContent,/Save as shapes/);
  assert.match(p.$('results').textContent,/Model: cpsam_20250102/);
  assert.ok(p.calls.stain.includes(4));
  assert.deepEqual(p.calls.setFrame.filter(v=>v!==undefined),[]);
});

test('stain download progress and a plain failure are visible in the full page', async () => {
  const p=page({runOverride:{kind:'stain',info:null,frames:0,progress:{done:0,total:4},
    stain:{model:'cpdino'},message:'Downloading the model…'}});
  await settle(25);
  assert.match(p.$('run-facts').textContent,/Downloading the model…/);
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'failed',
    message:'Could not download the model. Check the connection and try again.'}});
  await settle(20);
  assert.equal(p.$('run-error').hidden,false);
  assert.equal(p.$('run-error').textContent,'Could not download the model. Check the connection and try again.');
});

test('a new stain run shows Press Start and only the Start click requests segmentation', async () => {
  const p=page({runOverride:{kind:'stain',info:null,frames:0,state:'paused',paused:true,stain_started:false,
    progress:{done:0,total:0,revision:0},ack:{state:'paused'}}});
  await settle(25);
  assert.equal(p.$('run-toggle').textContent,'Start');
  assert.match(p.$('prior-hint').textContent,/Press Start to begin segmenting/);
  assert.equal(p.$('prior-hint').hidden,false);
  assert.deepEqual(p.calls.controls,[]);
  p.$('run-toggle').click();await settle(15);
  assert.deepEqual(p.calls.controls,['resume']);
});

for (const mode of ['detail', 'tiled']) test(`stain ${mode} tile and resolved outlines survive fit and pan with the stain picker selected`, async t => {
  const requests = [], polygon = {id:1,poly:[[10,10],[20,10],[20,20],[10,20],[10,10]]};
  t.mock.method(globalThis, 'fetch', async url => {
    const revision = Number(new URL(url, 'http://localhost').searchParams.get('revision'));
    requests.push(url);
    return {ok:true,arrayBuffer:async()=>encodeVectors({cells:[polygon]},
      {revision,resolved:revision===2,source:{cells:'Cellpose'}})};
  });
  const p = page({realStain:true,runOverride:{kind:'stain',mode,info:null,frames:0,
    progress:{done:1,total:2,revision:1},box:[0,0,100,100]}});
  await settle(25);
  t.after(()=>p.main().stainView.reset());
  p.$('run-kind').value = 'stain';   // Starting a run leaves the hidden picker's selection in place.
  const {priors,stage,state} = p.main();
  for (const resolved of [false,true]) {
    if (resolved) {
      p.socket().message({ev:'run',run:{...state.run,state:'done',
        progress:{done:2,total:2,revision:2,resolved:true,complete:true}}});
      await settle(20);
    }
    const vectors = priors.vectors;
    assert.equal(vectors.cells.count,1);
    assert.equal(vectors.resolved,resolved);
    assert.equal(p.snapshot().priors.drawn,false,'loading alone is not a paint acknowledgement');
    // PriorLayers owns this acknowledgement; its real paint path is covered in prior-layers.test.mjs.
    priors.shown = {vectors};
    assert.equal(p.snapshot().priors.drawn,true);
    vm.runInContext('fitView()',p.context);
    stage.camera.x += 2;
    for (const changed of stage.listeners) changed();
    await new Promise(resolve=>setTimeout(resolve,140));
    assert.equal(priors.vectors,vectors,'camera changes must not clear the run outlines');
    assert.equal(p.snapshot().priors.cells,1);
    assert.equal(p.snapshot().priors.drawn,true);
  }
  assert.equal(requests.length,2,'a covered viewport reuses the current tile or resolved vectors');
  priors.shown = {vectors:{...priors.vectors}};
  assert.equal(p.snapshot().priors.drawn,false,'a previous paint cannot acknowledge new vectors');
  priors.shown = {vectors:priors.vectors};
  state.run = {...state.run,progress:{...state.run.progress,revision:3}};
  assert.equal(p.snapshot().priors.drawn,false,'old tile outlines cannot acknowledge a new revision');
});

test('stain terminal updates disable Pause immediately, before the next paint batch', async () => {
  const p = page({runOverride:{kind:'stain',info:null,frames:0,paused:false,
    progress:{done:1,total:2,revision:1}}});
  await settle(25);
  const toggle = p.$('run-toggle'), classes = new Set(['primary']);
  toggle.classList.toggle = (name,on) => on ? classes.add(name) : classes.delete(name);
  assert.equal(toggle.disabled,false);
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'done',
    progress:{done:2,total:2,revision:3,resolved:true,complete:true}}});
  assert.equal(toggle.disabled,true);
  assert.equal(p.$('run-stop').disabled,true);
  assert.equal(classes.has('primary'),false,'an ended run must not advertise an active Pause action');
  await settle(20);
  assert.equal(toggle.disabled,true);
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'stopped',
    progress:{done:1,total:2,revision:4,resolved:true,complete:false}}});
  assert.equal(toggle.textContent,'Resume');
  assert.equal(toggle.disabled,false,'an unfinished stopped run can still resume');
});

for(const saveState of ['outdated','saved']) test(`stain results label an older save as outdated (${saveState})`,async()=>{
  const p=page({runOverride:{kind:'stain',info:null,frames:0,
    progress:{done:2,total:6,revision:3,resolved:false,complete:false}}});
  await settle(25);
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'done',
    progress:{done:6,total:6,revision:10,resolved:true,complete:true},
    zarr_save:{state:saveState,shapes_key:'cellpose_partial',table_key:null,cells:2,result_revision:3}}});
  await settle(20);
  assert.equal(p.$('results').hidden,false);
  assert.match(p.$('results').textContent,/Saved shapes are from an earlier result/);
  assert.doesNotMatch(p.$('results').textContent,/Saved as cellpose_partial/);
  p.socket().message({ev:'run',run:{...p.main().state.run,
    zarr_save:{state:'saved',shapes_key:'cellpose_full',table_key:null,cells:6,result_revision:10}}});
  await settle(20);
  assert.match(p.$('results').textContent,/Saved as cellpose_full/);
  assert.doesNotMatch(p.$('results').textContent,/earlier result/);
});

test('stain results offer saving completed shapes after a bookkeeping failure',async()=>{
  const p=page({runOverride:{kind:'stain',info:null,frames:0,
    progress:{done:2,total:6,revision:3,resolved:false,complete:false},outputs:[]}});
  await settle(25);
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'failed',
    progress:{done:6,total:6,revision:10,resolved:true,complete:true}}});
  await settle(20);
  assert.equal(p.$('results').hidden,false);
  assert.match(p.$('results').textContent,/Save as shapes/);
});


test('real tissue and area panels follow run detail in the full page and ignore older versions',async()=>{
  // The exclusions state and the run detail's `exclusions` carry the model's cells and those kept (I-C2).
  const p=page({realRoundtrip:true,runOverride:{kind:'stain',state:'paused',paused:true,ack:{state:'paused'},
    info:null,frames:0,stain:{model:'cpsam',image:'image',tissue_only:true},box:[0,0,100,100],
    tissue:{tiles_skipped:3},exclusions:{version:0,cells:7,kept:7,removed:0},progress:{done:1,total:1,revision:2,resolved:true},
    summary_model:{cells:7},summary:{cells:7}}});
  p.context.api.stainExclusions=async()=>({version:0,areas:[],removed:0,cells:7,kept:7});
  await settle(25);
  assert.equal(p.$('notice').textContent,'');assert.equal(p.main().tissueView.polygons.length,1);
  assert.match(p.$('run-facts').textContent,/3 tiles skipped outside the tissue/);
  assert.equal(p.$('stain-exclusions').hidden,false);
  // The panel's own words (stain-exclusions.mjs EXCLUSION_TEXT: "Keeps n of m cells", or every cell kept).
  const T=p.context.EXCLUSION_TEXT,first=p.$('stain-exclude-total').textContent;
  assert.equal(first,T.keptAll?T.keptAll.replace('{m}','7'):T.kept.replace('{n}','7').replace('{m}','7'));
  p.context.api.stainExclusions=async()=>({version:1,areas:[{id:'a',points:[[1,1],[90,1],[1,90]],cells:3}],removed:3,cells:7,kept:4});
  const before=p.main().state.run;
  p.socket().message({ev:'run',run:{...before,exclusions:{version:1,removed:3,count:1,cells:7,kept:4},summary:{cells:4}}});
  // They follow the newer detail: 4 of the model's 7 cells kept, 3 removed.
  await settle(10);const newer=p.$('stain-exclude-total').textContent;assert.equal(newer,'Keeps 4 of 7 cells');
  assert.equal(p.main().stainExclusions.data.areas.length,1);
  p.socket().message({ev:'run',run:before});await settle(10);
  assert.equal(p.main().state.run.exclusions.version,1);assert.equal(p.main().state.run.summary.cells,4);
  assert.equal(p.$('stain-exclude-total').textContent,newer,'an older detail changes nothing');
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'running',paused:false}});await settle(10);
  assert.equal(p.$('stain-exclusions').hidden,true);
});

// ---- the run picker: Start is never greyed out; a press says what is missing ------------------------------------------
test('the picker in the full page: a press of Start lists what is missing, and the snapshot says what Start needs and when it is busy', async () => {
  const p = page({ hook: true });
  await settle(20);
  const { state } = p.main(), start = p.$('start'), P = stain.PICKER_TEXT;
  p.$('run-kind').value = 'proseg';
  vm.runInContext('enterPick("s")', p.context);
  assert.deepEqual(plain(p.snapshot().picker), { kind: 'proseg', problems: [P.loading], preview: 0, boxes: null, busy: false }, 'loading: said, never greyed out');
  for (let i = 0; i < 40 && vm.runInContext('pickerLoading', p.context); i++) await settle(1);
  await settle(20);
  assert.equal(start.disabled, false);
  assert.deepEqual(plain(p.snapshot().picker), { kind: 'proseg', problems: [], preview: 0, boxes: null, busy: false });
  // No starting cells: the press names the field, outlines it and starts nothing.
  state.prior = { cells: null, nuclei: null };
  await start.click();
  assert.equal(p.calls.created, undefined);
  assert.deepEqual(plain(p.snapshot().picker.problems), ['prior-choice']);
  assert.ok(p.$('prior-choice').classList.contains(NEEDS_CLASS));
  assert.match(start.nextSibling.textContent, /Start from: choose where proseg starts its cells/);
  // Chosen again: a press starts the run, busy ("Starting…") until the engine answers; a second press says so.
  p.$('prior-choice').emit('change');
  const gate = deferred(), create = p.context.api.createRun;
  p.context.api.createRun = async (...args) => { await gate.promise; return create(...args); };
  const pressed = start.click();
  assert.equal(start.textContent, P.starting); assert.equal(p.snapshot().picker.busy, true);
  await start.click();
  assert.ok(start.nextSibling.textContent.includes(EXPLAIN_TEXT.busy));
  gate.resolve(); await pressed; await settle(20);
  assert.equal(p.calls.created.length, 4); assert.equal(state.mode, 'run'); assert.equal(p.snapshot().picker, null);
  assert.equal(start.getAttribute('aria-busy'), null);
  // New run without a sample says to open one, never greyed out.
  vm.runInContext('enterPick(null)', p.context);
  const newRun = p.$('new-run');
  assert.equal(newRun.disabled, false); await newRun.click();
  assert.ok(newRun.nextSibling.textContent.includes(P.openSample)); assert.equal(state.mode, 'empty');
});

test("the picker's Lasso in the full page: clicks place corners, a double-click closes them into the region and the view stays; Escape gives up the next", async () => {
  const p = page();
  await settle(20);
  const { state, stage } = p.main(), canvas = p.$('stage');
  p.$('run-kind').value = 'proseg';
  vm.runInContext('enterPick("s")', p.context);
  for (let i = 0; i < 40 && vm.runInContext('pickerLoading', p.context); i++) await settle(1);
  await settle(20);
  Object.assign(canvas, { setPointerCapture() {}, hasPointerCapture: () => true, releasePointerCapture() {} });
  let fits = 0; stage.fit = () => { fits++; };
  vm.runInContext('setRegionMode("polygon")', p.context);
  assert.match(p.$('region-help').textContent, /^Click to place corners or drag to draw; double-click, Enter or click the first point to close; /);
  const pointer = (type, x, y) => canvas.emit(type, { offsetX: x, offsetY: y, pointerId: 1, button: 0, shiftKey: false });
  const click = (x, y) => { pointer('pointerdown', x, y); pointer('pointerup', x, y); };
  for (const [x, y] of [[300, 300], [600, 300], [600, 600], [300, 600], [300, 600]]) click(x, y);
  canvas.emit('dblclick', { offsetX: 300, offsetY: 600, shiftKey: false });
  assert.deepEqual(plain(state.region), { kind: 'polygon', points: [[300, 300], [600, 300], [600, 600], [300, 600]] });
  assert.equal(fits, 0, 'the double-click closed the outline instead of fitting the view');
  // The page's own keys and the outline's share the document: Escape gives the next outline up, the region as it was.
  click(900, 900); click(1200, 900);
  for (const fn of p.context.document.listeners.get('keydown')) fn({ key: 'Escape', target: canvas, preventDefault() {}, metaKey: false, ctrlKey: false, altKey: false });
  assert.equal(vm.runInContext('lasso.active', p.context), false);
  assert.deepEqual(plain(state.region.points), [[300, 300], [600, 300], [600, 600], [300, 600]]);
  assert.equal(p.$('notice').hidden, true, p.$('notice').textContent);
  // This file strips the imports and hands the page lasso.mjs itself: the page must import it.
  assert.match(readFileSync(new URL('../src/main.js', import.meta.url), 'utf8'), /^import \{ LassoDraft, mergedMoves \} from '\.\/lasso\.mjs';$/m);
});

/** The full page's picker on Training patches (Model outlines) for a sample with one image, complete as it opens; `preview`
 *  answers for the engine's boxes. Afterwards (also when the test fails) the page leaves the picker and its timers, so it
 *  never keeps asking for models every 15 s or for boxes, and the test file ends at once. */
async function patchesPage(t, preview) {
  const p = page({ hook: true });
  t.after(() => vm.runInContext('loadStainModels.ticket = (loadStainModels.ticket || 0) + 1; clearTimeout(loadStainModels.timer); ' +
    'enterPick(null); clearTimeout(noticeTimer)', p.context));
  await settle(20);
  const { state } = p.main(), api = p.context.api, sample = api.sample, fetch = p.context.fetch;
  // The picker shows the chosen image channel (2 × 2 pixels here).
  p.context.fetch = async (url, o) => /\.u16\?/.test(url) ? { ok: true, arrayBuffer: async () => new Uint16Array(4).buffer,
    headers: { get: () => JSON.stringify({ x0: 0, y0: 0, x1: 2000, y1: 2000, width: 2, height: 2 }) } } : fetch(url, o);
  state.health = { ...state.health, stain: { available: true } };
  api.sample = async id => ({ ...(await sample(id)), images: [{ key: 'im', channels: ['DAPI'], um_per_px: 1 }], image_key: 'im' });
  api.stainModels = async () => ({ models: [{ name: 'cpsam', kind: 'builtin' }] });
  api.previewPatches = preview;
  api.createStain = async (...args) => { p.calls.stainCreated = args; return { id: 'r1' }; };
  // Stain segmentation chosen (the page opened a proseg run), with its Training patches tool open, as New run from an item.
  vm.runInContext("setTool('stain')", p.context);
  p.$('run-kind').value = 'patches'; p.$('patch-start-from').value = 'model';
  vm.runInContext('enterPick("s")', p.context);
  for (let i = 0; i < 40 && vm.runInContext('pickerLoading', p.context); i++) await settle(1);
  await settle(20);
  return p;
}
const later = ms => new Promise(resolve => setTimeout(resolve, ms));

/** Every word a user can reach on the page: each node's own text, an option's value, and the attributes that are read or
 *  shown (titles, labels, placeholders), never an element's id, class or data- name. */
function reachable(root) {
  const texts = [];
  for (const node of root.all()) {
    texts.push(node.text || '', node.title || '', node.placeholder || '', node.label || '', node.alt || '', node.tagName === 'OPTION' ? node.value : '');
    for (const [key, value] of node.attributes || []) if (!['id', 'class'].includes(key) && !key.startsWith('data-')) texts.push(value);
  }
  return texts.filter(Boolean);
}
/** The whole page booted on an engine whose health lists `methods` for stain segmentation, then Stain segmentation's
 *  picker opened on a sample with one image, its models listed (the methods asked for, in `asked`). */
async function stainToolPage(t, methods) {
  const p = page({ hook: true, apiOverride: { health: async () => ({ proseg: true, one_run: true, stain: { available: true, methods } }),
    runs: async () => [] } });
  t.after(() => vm.runInContext('loadStainModels.ticket = (loadStainModels.ticket || 0) + 1; clearTimeout(loadStainModels.timer); ' +
    'enterPick(null); clearTimeout(noticeTimer)', p.context));
  await settle(20);
  const api = p.context.api, sample = api.sample, fetch = p.context.fetch, asked = [];
  p.context.fetch = async (url, o) => /\.u16\?/.test(url) ? { ok: true, arrayBuffer: async () => new Uint16Array(4).buffer,
    headers: { get: () => JSON.stringify({ x0: 0, y0: 0, x1: 2000, y1: 2000, width: 2, height: 2 }) } } : fetch(url, o);
  api.sample = async id => ({ ...(await sample(id)), images: [{ key: 'im', channels: ['DAPI', 'Boundary'], um_per_px: 1 }], image_key: 'im' });
  api.stainModels = async method => { asked.push(method); return { models: [{ name: 'cpsam', kind: 'builtin' }] }; };
  // The channel list's chosen options, as a browser gives them (InstanSeg's channels are read from it).
  Object.defineProperty(p.$('stain-channels'), 'selectedOptions', { get() { return this.children.filter(o => o.selected); } });
  vm.runInContext("setTool('stain')", p.context);
  p.$('run-kind').value = 'stain';
  vm.runInContext('enterPick("s")', p.context);
  for (let i = 0; i < 40 && vm.runInContext('pickerLoading', p.context); i++) await settle(1);
  await settle(20);
  return { p, asked };
}

test('InstanSeg is dormant: the whole page offers Cellpose alone, with no Method row and no word of InstanSeg anywhere', async t => {
  const { p, asked } = await stainToolPage(t, { cellpose: true });
  const { state } = p.main();
  assert.equal(state.mode, 'pick'); assert.equal(p.$('run-kind').value, 'stain'); assert.equal(p.$('stain-picker').hidden, false);
  // The markup's own option (its words come from stainCopy, which this page's fake document cannot select for).
  assert.deepEqual(p.$('stain-method').children.map(o => o.value), ['cellpose']);
  assert.equal(p.$('stain-method').parentElement.hidden, true, 'one method: no Method row');
  assert.equal(p.$('stain-instanseg-options').hidden, true, 'no Channels, Find or Match cells and nuclei');
  assert.deepEqual([...new Set(asked)], ['cellpose'], 'only Cellpose\'s models are asked for');
  const words = reachable(p.dom.document.body);
  assert.ok(words.some(text => /Use settings from Cellpose/.test(text)) && words.some(text => /Proseg Live/.test(text)),
    'the page\'s words are read');
  assert.deepEqual(words.filter(text => /instanseg/i.test(text)), []);
  assert.match(p.$('tool-stain').textContent, /with Cellpose, clean them up/);     // the first screen's card too
});

test('the engine\'s internal switch on, the page offers InstanSeg again beside Cellpose', async t => {
  const { p, asked } = await stainToolPage(t, { cellpose: true, instanseg: true });
  assert.deepEqual(p.$('stain-method').children.map(o => [o.value, o.textContent]), [['cellpose', 'Cellpose'], ['instanseg', 'InstanSeg']]);
  assert.equal(p.$('stain-method').parentElement.hidden, false, 'a choice: the Method row shows');
  assert.equal(p.$('stain-method').value, 'cellpose');
  p.$('stain-method').value = 'instanseg'; p.$('stain-method').emit('change'); await settle(20);
  assert.equal(p.$('stain-instanseg-options').hidden, false);
  assert.deepEqual([...new Set(asked)], ['cellpose', 'instanseg']);
});

test('Training patches in the full page: the boxes count in the snapshot once they match the picker, Start sends them, and opening the item clears them', async t => {
  const asked = [], boxes = [[0, 0, 64, 64], [64, 0, 128, 64]];
  const p = await patchesPage(t, async (...args) => { asked.push(args);
    return { patches: boxes.map((box, i) => ({ id: i + 1, box })), run_pixel_to_sample: [850, 850, 1, 1], message: '', extent: [300, 300] }; });
  const { state } = p.main();
  assert.deepEqual(plain(p.snapshot().picker), { kind: 'patches', problems: [], preview: 0, boxes: null, busy: false }, 'complete: the boxes are on their way');
  await later(450); await settle(5);
  assert.equal(asked.length, 1, 'asked once, 400 ms after the picker was complete');
  assert.deepEqual(plain(asked[0].slice(0, 2)), ['s', { kind: 'box', box: [850, 850, 1150, 1150] }]);
  assert.deepEqual(plain(p.snapshot().picker), { kind: 'patches', problems: [], preview: 2, boxes, busy: false });
  // A change: its boxes are due. Those on the image stay, but count only once the change has its own (the harness waits).
  p.$('patch-count').value = '5'; p.$('patch-count').emit('input');
  assert.deepEqual(plain(p.snapshot().picker), { kind: 'patches', problems: [], preview: 0, boxes, busy: false });
  await later(450); await settle(5);
  assert.equal(asked.length, 2); assert.equal(asked[1][2].count, 5); assert.equal(p.snapshot().picker.preview, 2);
  await p.$('start').click(); await settle(20);
  assert.deepEqual(plain(p.calls.stainCreated[2].boxes), boxes); assert.equal(p.calls.stainCreated[3], 'patches');
  assert.equal(state.mode, 'run'); assert.equal(p.calls.previewClears, 1, 'the item opened: the picker\'s boxes go');
  assert.equal(p.$('notice').hidden, true, p.$('notice').textContent);
});

test('Training patches in the full page: another run opened while Start waits for its boxes creates nothing, leaves Start free and asks for nothing more', async t => {
  const asked = [];
  const p = await patchesPage(t, (sample, region, stain, signal) => new Promise((resolve, reject) => {
    asked.push({ signal, resolve });
    signal.addEventListener('abort', () => reject(Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' })));
  }));
  const { state } = p.main(), api = p.context.api, start = p.$('start');
  // Complete, its boxes due in 400 ms: Start pressed meanwhile asks for them at once and waits.
  assert.equal(vm.runInContext('patchPreview.timer > 0', p.context), true); assert.equal(asked.length, 0);
  const pressed = start.click();
  assert.equal(asked.length, 1); assert.equal(start.getAttribute('aria-busy'), 'true');
  // Another run opened (from the runs list) before they arrive, and slow to load.
  const slow = deferred(), run = api.run;
  api.run = async id => { await slow.promise; return run(id); };
  const opened = vm.runInContext('openRun("r1")', p.context);
  assert.equal(asked[0].signal.aborted, true, 'the request on its way is cancelled at once');
  await pressed;
  assert.equal(p.calls.stainCreated, undefined, 'nothing is created');
  assert.equal(start.getAttribute('aria-busy'), null, 'Start is not left busy'); assert.equal(start.textContent, STAIN_PATCH_TEXT.start);
  await later(450);
  assert.equal(asked.length, 1, 'nothing more is asked for while the run loads');
  slow.resolve(); await opened; await settle(20);
  assert.equal(state.mode, 'run'); assert.equal(state.run.id, 'r1');
  assert.equal(asked.length, 1); assert.equal(p.calls.stainCreated, undefined);
});

test('Training patches in the full page: opening a run while boxes are due cancels them; nothing is asked for while it loads', async t => {
  const asked = [];
  const p = await patchesPage(t, async (...args) => { asked.push(args); return { patches: [], run_pixel_to_sample: [850, 850, 1, 1], message: '' }; });
  const api = p.context.api, slow = deferred(), run = api.run;
  assert.equal(vm.runInContext('patchPreview.timer > 0', p.context), true, 'due in 400 ms');
  api.run = async id => { await slow.promise; return run(id); };
  const opened = vm.runInContext('openRun("r1")', p.context);
  assert.equal(vm.runInContext('patchPreview.timer', p.context), 0, 'cleared as the run starts opening');
  await later(450);
  assert.equal(asked.length, 0, 'no request after');
  slow.resolve(); await opened; await settle(20);
  assert.equal(p.main().state.mode, 'run'); assert.equal(asked.length, 0);
});

test('Training patches in the full page: a run that cannot be opened leaves the picker\'s boxes; what was due then is asked for by Start', async t => {
  const asked = [], boxes = [[0, 0, 64, 64], [64, 0, 128, 64]];
  const p = await patchesPage(t, async (...args) => { asked.push(args);
    return { patches: boxes.slice(0, args[2].count).map((box, i) => ({ id: i + 1, box })), run_pixel_to_sample: [850, 850, 1, 1], message: '', extent: [300, 300] }; });
  const { state } = p.main(), api = p.context.api, run = api.run;
  await later(450); await settle(5);
  assert.equal(asked.length, 1); assert.deepEqual(plain(p.snapshot().picker.boxes), boxes);
  // Another run fails to open (it is gone): the picker stays as it was, boxes and all.
  api.run = async () => { throw Object.assign(new Error('Unknown run'), { status: 404 }); };
  await assert.rejects(vm.runInContext('openRun("gone")', p.context), /Unknown run/);
  assert.equal(state.mode, 'pick'); assert.equal(p.calls.previewClears, undefined, 'the boxes stay');
  assert.deepEqual(plain(p.snapshot().picker), { kind: 'patches', problems: [], preview: 2, boxes, busy: false });
  // Again with a change due (one patch): cancelled as the run starts opening, the boxes stay, and nothing is asked for by
  // itself; Start asks for it at once and makes the item with what comes.
  p.$('patch-count').value = '1'; p.$('patch-count').emit('input');
  assert.equal(vm.runInContext('patchPreview.timer > 0', p.context), true);
  await assert.rejects(vm.runInContext('openRun("gone")', p.context), /Unknown run/);
  assert.equal(vm.runInContext('patchPreview.timer', p.context), 0); assert.deepEqual(plain(p.snapshot().picker.boxes), boxes);
  await later(450); assert.equal(asked.length, 1);
  api.run = run;
  await p.$('start').click(); await settle(20);
  assert.equal(asked.length, 2); assert.equal(asked[1][2].count, 1);
  assert.deepEqual(plain(p.calls.stainCreated[2].boxes), [boxes[0]]); assert.equal(state.mode, 'run');
  assert.equal(p.calls.previewClears, 1, 'the item opened: now they go');
});

// ---- the app shell: the tool in the header, home, Close and the window's ✕ ---------------------------------------------
test('a run opened names its tool in the header and sets the picker\'s run type; the brand leaves the run and goes home', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });            // the panels' batch waits on no timer after the test
  const p = page();
  await settle(20);
  const { state } = p.main();
  p.$('run-kind').value = 'stain';                             // left from a Stain segmentation picker before
  vm.runInContext('openRun("r1")', p.context); await settle(20);
  assert.deepEqual([state.mode, state.tool, p.$('tool-name').textContent, p.$('tool-name').hidden, p.$('run-kind').value],
    ['run', 'proseg', 'Proseg Live', false, 'proseg'], 'New run from here is a Proseg Live picker');
  p.$('home').click(); await discardRun(p); await settle(5);
  assert.deepEqual([state.mode, state.tool, state.run, p.$('empty').hidden, p.$('tool-choice').hidden, p.$('dock').hidden, p.$('tool-name').hidden],
    ['empty', null, null, false, false, true, true], 'the first screen\'s two cards');
  assert.deepEqual(p.calls.controls, [], 'the engine owns leaving');
  const stained = page({ runOverride: { kind: 'patches', state: 'done', info: null, frames: 0, progress: { done: 0, revision: 0 },
    box: [0, 0, 100, 100], stain: { method: 'cellpose', model: 'cpsam' }, start_from: 'model' } });
  await settle(25);
  assert.equal(stained.main().state.mode, 'empty', 'a done item is not opened by itself');
  stained.context.api.frames = async () => ({ from: 0, frames: [] });   // an item has no iterations
  vm.runInContext('openRun("r1")', stained.context); await settle(25);   // explicit run URL
  assert.deepEqual([stained.main().state.tool, stained.$('tool-name').textContent, stained.$('run-kind').value],
    ['stain', 'Stain segmentation', 'stain'], 'a Training patches item opens in Stain segmentation; New run stays there');
});

test('a run opened tells Layers its sample, whose image\'s contrast limits are kept for it', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });            // the panels' batch waits on no timer after the test
  const p = page();
  await settle(20);
  assert.equal(p.main().state.run.sample, 's'); assert.equal(p.main().layerPanel.sample, 's');
});

test('an engine that answers none of the first three requests: after 20 s the first screen says so, with Reload', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = page({ apiOverride: { health: () => new Promise(() => {}) } });   // never answers
  await settle(20);
  assert.deepEqual([p.main().state.mode, p.$('engine-down').hidden], ['empty', true], 'the cards while it waits');
  t.mock.timers.tick(19999); await settle(5);
  assert.equal(p.$('engine-down').hidden, true);
  t.mock.timers.tick(1); await settle(20);
  assert.deepEqual([p.$('engine-down').hidden, p.$('engine-down-title').textContent, p.$('engine-reload').textContent, p.$('tool-choice').hidden],
    [false, 'Cannot reach the engine', 'Reload', true]);
  assert.match(p.$('engine-down-why').textContent, /did not answer within 20 seconds/);
  assert.equal(p.$('notice').hidden, true, 'not a passing notice');
});

/** The page with a run of `state` open, its Close and the engine's answer recorded: the window's close() and POST /closing.
 *  Its timers are the test's (`t`): the check that the window did close is never left running after the test. */
async function closingPage(t, state, { hook = false } = {}) {
  if (!closingPage.mocked?.has(t)) { t.mock.timers.enable({ apis: ['setTimeout'] }); (closingPage.mocked ??= new WeakSet()).add(t); }
  const p = page({ runOverride: { state }, hook });
  await settle(20);
  const answers = [], revisions = [], win = p.context.window;
  p.context.api.closing = async (answer, options = {}) => { answers.push([answer, !!options.beacon, options.runs ?? [], !!options.end]);
    revisions.push(options.revision); return { answer }; };
  win.closes = 0; win.closed = false; win.close = () => { win.closes++; win.closed = true; };
  const fire = (type, event = {}) => { for (const fn of win.listeners.get(type) || []) fn(event); return event; };
  return { p, answers, revisions, win, fire };
}

test('Close asks for unsaved results and waits for gone before closing', async t => {
  const {p,win}=await closingPage(t,'running');
  p.$('close-app').click();await settle();assert.equal(p.$('leave-popup').open,true);assert.equal(win.closes,0);
  await p.$('leave-save').click();assert.deepEqual(p.calls.leaves,[['r1','add']]);assert.equal(win.closes,0);
  p.socket().message({ev:'gone',run:'r1',reason:'left'});await settle();assert.equal(win.closes,1);
});
test('Close also asks for a finished unsaved run; Stay keeps the view',async t=>{
  const {p,win}=await closingPage(t,'done');
  vm.runInContext('openRun("r1")',p.context);await settle(20);
  p.$('close-app').click();await settle();assert.equal(p.$('leave-popup').open,true);
  await p.$('leave-stay').click();assert.equal(p.main().state.mode,'run');assert.equal(win.closes,0);
});
test('window close without an answer sends discard, for the run it is about, with a revision',async t=>{
  const {p,answers,revisions,win}=await closingPage(t,'running');
  const event={preventDefault(){this.prevented=true;}};
  for(const fn of win.listeners.get('beforeunload'))fn(event);await settle();
  assert.equal(event.prevented,true);assert.equal(p.$('leave-popup').open,true);
  assert.equal(p.$('leave-title').textContent,'Close SoPA companion?');
  for(const fn of win.listeners.get('pagehide'))fn();assert.deepEqual(plain(answers),[['discard',true,['r1'],false]]);
  assert.ok(Number.isInteger(revisions[0])&&revisions[0]>0);
});

test('Not now in the Save dialog after Finish & save posts leave stay, so closing no longer saves it whatever the answer; any other run posts nothing',async()=>{
  const p=page({runOverride:{state:'done',finish_requested:true,zarr_save:null},apiOverride:{health:async()=>({one_run:true,proseg:true})}});
  await settle(30);
  const sent=[],{state}=p.main();
  p.context.api.leave=async(id,answer,revision)=>{sent.push([id,answer,Number.isInteger(revision)]);return {...state.run,save_cancelled:true};};
  assert.equal(state.mode,'run');
  p.calls.zarrSave.onCancel({...state.run});await settle();
  assert.deepEqual(sent,[['r1','stay',true]]);assert.equal(state.run.save_cancelled,true,'the engine\'s answer is the run from then on');
  const run={...state.run,save_cancelled:false};
  for(const other of [{...run,save_cancelled:true},{...run,finish_requested:false},{...run,zarr_save:{state:'failed',message:'Disk full'}},
    {...run,leaving:{answer:'add',stage:'writing'}},{...run,state:'paused'}])p.calls.zarrSave.onCancel(other);
  await settle();assert.equal(sent.length,1,'only a Finish & save still counting as answered is taken back');
});
test('Close with a run whose Finish & save counts as answered: named as being saved whatever the answer, and Discard and close saves it too',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const p=page({runOverride:{state:'done',finish_requested:true,zarr_save:null},apiOverride:{health:async()=>({one_run:true,proseg:true})}});
  await settle(30);
  const win=p.context.window;win.closes=0;win.close=()=>{win.closes++;win.closed=true;};
  p.$('close-app').click();await settle();
  assert.equal(p.$('leave-popup').open,true);assert.equal(p.$('leave-lead').textContent,'Being saved already, whatever you choose:');
  assert.equal(p.$('leave-help').textContent,'Finish & save was pressed, so it is saved to Sample under new names before the app closes.');
  await p.$('leave-discard').click();assert.deepEqual(p.calls.leaves,[['r1','add']]);assert.equal(win.closes,0);
  p.socket().message({ev:'gone',run:'r1',reason:'left'});await settle();assert.equal(win.closes,1);
});
test('a window showing no run holds nothing and answers nothing for another window\'s run (behind the older runs\' prompt); once it shows the run, it does',async t=>{
  const p=page({apiOverride:{health:async()=>({one_run:true,proseg:true,older:{runs:[{id:'old-a',state:'done',sample_name:'Sample A',can_add:true}]}}),
    olderRuns:async()=>{throw Object.assign(new Error('Not Found'),{status:404});},answerOlder:async()=>({})}});
  await settle(30);
  const prompt=vm.runInContext('olderPrompt',p.context);t.after(()=>prompt.finish());
  const answers=[];p.context.api.closing=async(answer,options={})=>{answers.push([answer,options.runs]);return {};};
  assert.equal(prompt.open,true);assert.equal(p.main().state.run,null);assert.equal(p.main().state.runs.length,1,'a run of another window, still sampling');
  const win=p.context.window,fire=type=>{const event={preventDefault(){this.prevented=true;}};for(const fn of win.listeners.get(type)||[])fn(event);return event;};
  assert.equal(fire('beforeunload').prevented,undefined,'nothing held');fire('pagehide');
  assert.deepEqual(answers,[],'no answer for a run this window never showed');
  prompt.dialog.emit('cancel');await settle(30);
  assert.equal(p.main().state.run?.id,'r1');assert.equal(fire('beforeunload').prevented,true);fire('pagehide');
  assert.deepEqual(plain(answers),[['discard',['r1']]]);
});
test('the start notice says how many runs of earlier versions were removed, also with no other leftover',async()=>{
  const p=page({apiOverride:{health:async()=>({one_run:true,proseg:true,leftovers:{stamp:`older-removed-${Math.random()}`,runs:[],older_removed:2}}),runs:async()=>[]}});
  await settle(30);
  assert.equal(p.$('notice').textContent,'2 runs from earlier versions were removed: they were saved already or had nothing to save.');
});

// ---- a notebook's session (API mode) ---------------------------------------------------------------------------------------
const sessionOf = (sample = {}, over = {}) => ({ mode: 'api', id: '1f3a9c2e', sample: { id: 's', name: 'SMP02', backed: true, zarr_name: 'SMP02.zarr',
  view: 'in_place', state: 'ready', progress: { done: 0, total: 0 }, message: null, memory_only: [], tools: ['proseg', 'stain'], ...sample },
  tool: null, writes: 'zarr', notebook: { busy: false }, patches: false, link: false, state: 'running', ...over });
test('a notebook\'s session: the cards open its sample (no Open sample, Recent or ‹ Both tools), the header names it, the first screen shows it loading; tool= opens its picker', async () => {
  const p = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true,
    session: sessionOf({ state: 'preparing', progress: { done: 2.1e9, total: 6.0e9 } }, { tool: 'proseg' }) }), runs: async () => [], samples: async () => [] } });
  await settle(30);
  const { state } = p.main();
  assert.equal(state.mode, 'empty'); assert.equal(p.$('tool-choice').hidden, false);
  assert.equal(p.$('tool-choose').textContent, 'Choose a tool for SMP02.');
  // Loading: the static spinner, the words and a bar of the part prepared.
  assert.equal(p.$('session-status').textContent, 'Preparing your SpatialData for the app: 2.1 of 6.0 GB'); assert.equal(p.$('session-status').hidden, false);
  assert.equal(p.$('session-spinner').hidden, false); assert.equal(p.$('session-progress').hidden, false); assert.equal(p.$('session-progress').value, 0.35);
  for (const id of ['open-sample', 'empty-open-sample', 'tool-back', 'recent']) assert.equal(p.$(id).hidden, true, id);
  assert.equal(p.$('patch-tool').hidden, false, 'Training patches in every session: each export asks where it goes');
  assert.equal(p.$('sample-name').textContent, 'SMP02'); assert.equal(p.$('sample-name').title, 'From your notebook; results go into this object and SMP02.zarr');
  assert.equal(p.context.document.title, 'SoPA companion · SMP02');
  // Opening (no size to show): the spinner and words only.
  let asked = 0; const sample = p.context.api.sample; p.context.api.sample = async id => { asked++; return sample(id); };
  p.socket().message({ ev: 'session', ...sessionOf({ state: 'opening' }, { tool: 'proseg' }) }); await settle();
  assert.equal(p.$('session-status').textContent, 'Opening SMP02…'); assert.equal(p.$('session-spinner').hidden, false); assert.equal(p.$('session-progress').hidden, true);
  assert.equal(state.mode, 'empty'); assert.equal(asked, 0);
  // Ready (the session's push): tool= opens that tool's picker on the bound sample, once.
  p.socket().message({ ev: 'session', ...sessionOf({}, { tool: 'proseg' }) }); await settle(30);
  assert.equal(state.mode, 'pick'); assert.equal(state.sample.id, 's'); assert.equal(state.tool, 'proseg'); assert.equal(asked, 1);
  assert.equal(p.$('session-status').hidden, true);
  assert.equal(p.$('sample-name').title, 'From your notebook; results go into this object and SMP02.zarr', 'never a path, also in the picker');
  // The brand: the cards again, still naming the sample; a card opens its picker at once now.
  await p.$('home').click(); await settle(10); assert.equal(state.mode, 'empty'); assert.equal(p.$('sample-name').textContent, 'SMP02');
  p.$('tool-stain').click(); await settle(30); assert.equal(state.mode, 'pick'); assert.equal(p.$('run-kind').value, 'stain');
});
test('in a session a card pressed while the SpatialData loads waits for it, loading, and opens by itself once it is ready (over tool=); another card takes its place', async () => {
  const p = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf({ state: 'opening' }, { tool: 'proseg' }) }),
    runs: async () => [], samples: async () => [] } });
  await settle(30);
  const { state } = p.main(), busy = id => p.$(id).getAttribute('aria-busy') === 'true';
  let asked = 0; const sample = p.context.api.sample; p.context.api.sample = async id => { asked++; return sample(id); };
  p.$('tool-proseg').click(); await settle();
  assert.equal(state.mode, 'empty'); assert.equal(asked, 0);
  assert.equal(busy('tool-proseg'), true); assert.equal(p.$('tool-proseg-wait').hidden, false);
  assert.equal(p.$('tool-proseg-wait-text').textContent, 'Opens when SMP02 is ready');
  assert.doesNotMatch(p.$('tool-proseg').nextSibling?.textContent ?? '', /Check these first|still being opened/, 'the press is taken, not explained');
  // The other card takes its place.
  p.$('tool-stain').click(); await settle();
  assert.equal(busy('tool-proseg'), false); assert.equal(p.$('tool-proseg-wait').hidden, true);
  assert.equal(busy('tool-stain'), true); assert.equal(p.$('tool-stain-wait-text').textContent, 'Opens when SMP02 is ready');
  // Still loading (the preparing push of another session's run): still waiting, nothing opens.
  p.socket().message({ ev: 'session', ...sessionOf({ state: 'opening', name: null }, { tool: 'proseg' }) }); await settle();
  assert.equal(busy('tool-stain'), true); assert.equal(p.$('tool-stain-wait-text').textContent, 'Opens when your SpatialData is ready');
  assert.equal(state.mode, 'empty'); assert.equal(asked, 0);
  // Ready: the card pressed opens its picker, not tool='proseg'; the waiting ends.
  p.socket().message({ ev: 'session', ...sessionOf({}, { tool: 'proseg' }) }); await settle(30);
  assert.equal(state.mode, 'pick'); assert.equal(state.tool, 'stain'); assert.equal(p.$('run-kind').value, 'stain'); assert.equal(asked, 1);
  assert.equal(busy('tool-stain'), false); assert.equal(p.$('tool-stain-wait').hidden, true);
  // A later push opens nothing more.
  p.socket().message({ ev: 'session', ...sessionOf({}, { tool: 'proseg', notebook: { busy: true } }) }); await settle(30);
  assert.equal(state.tool, 'stain'); assert.equal(asked, 1);
});
test('in a session a SpatialData that could not be opened ends the waiting: the reason in place of the spinner, and a card\'s press says it', async () => {
  const p = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf({ state: 'opening' }) }),
    runs: async () => [], samples: async () => [] } });
  await settle(30);
  const { state } = p.main();
  let asked = 0; const sample = p.context.api.sample; p.context.api.sample = async id => { asked++; return sample(id); };
  p.$('tool-stain').click(); await settle();
  assert.equal(p.$('tool-stain').getAttribute('aria-busy'), 'true');
  p.socket().message({ ev: 'session', ...sessionOf({ state: 'failed', message: 'No transcripts and no image the app can place' }) }); await settle(30);
  assert.equal(p.$('session-status').textContent, 'Could not open your SpatialData: No transcripts and no image the app can place');
  assert.equal(p.$('session-status').dataset.state, 'failed'); assert.equal(p.$('session-spinner').hidden, true); assert.equal(p.$('session-progress').hidden, true);
  assert.equal(p.$('tool-stain').getAttribute('aria-busy'), null); assert.equal(p.$('tool-stain-wait').hidden, true);
  assert.equal(state.mode, 'empty'); assert.equal(asked, 0);
  p.$('tool-stain').click(); await settle();
  assert.match(p.$('tool-stain').nextSibling.textContent, /Could not open your SpatialData: No transcripts/);
  assert.equal(p.$('tool-stain').getAttribute('aria-busy'), null); assert.equal(asked, 0);
});
test('Close in a session: a run not added yet is asked about in the session\'s words; the answer ends the session (POST /closing, end), then its ended screen with no Reload', async () => {
  const p = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf() }) } });
  await settle(30);
  const { state } = p.main(), answers = [];
  p.context.api.closing = async (answer, options = {}) => { answers.push([answer, !!options.beacon, options.runs ?? [], !!options.end, Number.isInteger(options.revision)]); return {}; };
  assert.equal(state.mode, 'run');
  await p.$('close-app').click(); await settle();
  assert.equal(p.$('leave-popup').open, true); assert.equal(p.$('leave-title').textContent, 'Close SoPA companion?');
  assert.equal(p.$('leave-lead').textContent, 'This run is not added to your SpatialData yet:');
  assert.equal(p.$('leave-save').textContent, 'Add to your SpatialData and close'); assert.equal(p.$('leave-stay').textContent, 'Keep working');
  await p.$('leave-stay').click(); assert.equal(p.$('leave-popup').open, false); assert.deepEqual(answers, [], 'Keep working asks the engine nothing');
  await p.$('close-app').click(); await settle(); await p.$('leave-save').click(); await settle(10);
  assert.deepEqual(plain(answers), [['add', false, ['r1'], true, true]]); assert.equal(p.calls.leaves, undefined, 'the engine adds it as the session closes');
  assert.equal(p.$('engine-down').hidden, false); assert.equal(p.$('engine-down-title').textContent, 'Closing…');
  // The engine's push, then its socket closing: the session has ended.
  p.socket().message({ ev: 'session', ...sessionOf({}, { state: 'ending' }) }); p.socket().connection(false); await settle(10);
  assert.equal(p.$('engine-down-title').textContent, 'Session ended');
  assert.equal(p.$('engine-down-why').textContent, 'This session has ended. You can close this tab, and start a new one from your notebook.');
  assert.equal(p.$('engine-reload').hidden, true); assert.equal(state.run, null); assert.equal(state.mode, 'empty');
  // The window going now: the session has its answer already; nothing holds the window and no beacon goes.
  const event = { preventDefault() { this.prevented = true; } }, win = p.context.window;
  for (const fn of win.listeners.get('beforeunload')) fn(event);
  for (const fn of win.listeners.get('pagehide')) fn();
  assert.equal(event.prevented, undefined); assert.deepEqual(plain(answers), [['add', false, ['r1'], true, true]], 'no beacon after the session closed');
});
test('in a session every other way of leaving asks in its words and goes through the leave, never ending the session', async () => {
  const p = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf() }) } });
  await settle(30);
  const closings = []; p.context.api.closing = async (...args) => { closings.push(args); return {}; };
  const next = p.$('new-run').click(); await settle();
  assert.equal(p.$('leave-title').textContent, 'Add this run to your SpatialData?');
  assert.equal(p.$('leave-save').textContent, 'Add to your SpatialData');
  assert.equal(p.$('leave-help').textContent, 'Add finishes it, then adds its cells under new names. Discard deletes it.');
  await p.$('leave-discard').click(); assert.deepEqual(p.calls.leaves, [['r1', 'discard']]); assert.deepEqual(closings, []);
  p.socket().message({ ev: 'gone', run: 'r1', reason: 'left' }); await next; await settle(20);
  assert.equal(p.main().state.mode, 'pick'); assert.equal(p.$('engine-down').hidden, true);
});
test('in a session the window\'s ✕ during a leave keeps that leave\'s prompt; its Stay ends the leave, and New run asks again (never stuck)', async () => {
  const p = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf() }) } });
  await settle(30);
  const closings = []; p.context.api.closing = async (...args) => { closings.push(args); return {}; };
  const next = p.$('new-run').click(); await settle();
  assert.equal(p.$('leave-title').textContent, 'Add this run to your SpatialData?');
  const win = p.context.window, event = { preventDefault() { this.prevented = true; } };
  for (const fn of win.listeners.get('beforeunload')) fn(event); await settle();
  assert.equal(event.prevented, true, 'the browser holds the window');
  assert.equal(p.$('leave-title').textContent, 'Add this run to your SpatialData?', 'the leave keeps its own prompt');
  assert.equal(p.$('leave-stay').textContent, 'Stay');
  await p.$('leave-stay').click(); await next; await settle();
  assert.equal(p.$('leave-popup').open, false); assert.equal(p.main().state.mode, 'run'); assert.deepEqual(closings, []);
  const again = p.$('new-run').click(); await settle(); assert.equal(p.$('leave-popup').open, true, 'New run asks again');
  await p.$('leave-discard').click(); p.socket().message({ ev: 'gone', run: 'r1', reason: 'left' }); await again; await settle(20);
  assert.equal(p.main().state.mode, 'pick');
});
test('Close in a session with nothing pending ends it at once; an engine gone without its push shows the ended screen', async () => {
  const p = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf() }), runs: async () => [] } });
  await settle(30);
  const answers = []; p.context.api.closing = async (answer, options = {}) => { answers.push([answer, options.runs ?? [], !!options.end]); return {}; };
  await p.$('close-app').click(); await settle(10);
  assert.deepEqual(plain(answers), [['discard', [], true]]); assert.equal(p.$('leave-popup').open, false);
  const q = page({ apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf() }), runs: async () => [] } });
  await settle(30);
  q.context.api.health = async () => { throw new TypeError('Failed to fetch'); };
  q.socket().connection(false); await settle(10);
  assert.equal(q.$('engine-down').hidden, false); assert.equal(q.$('engine-down-title').textContent, 'Session ended'); assert.equal(q.$('engine-reload').hidden, true);
  // On the desktop a closed socket only reconnects.
  const d = page({ apiOverride: { runs: async () => [] } }); await settle(30);
  d.context.api.health = async () => { throw new TypeError('Failed to fetch'); }; d.socket().connection(false); await settle(10);
  assert.equal(d.$('engine-down').hidden, true);
});

async function proceedPage(extra={}) {
  const saved={id:'r1',kind:'stain',state:'done',sample:'s',sample_name:'Sample',sample_kind:'zarr',sample_tools:['stain','proseg'],
    box:[0,0,100,100],region:{kind:'box',box:[0,0,100,100]},info:null,frames:0,
    stain:{method:'cellpose',model:'cpsam'},progress:{done:4,total:4,complete:true,resolved:true,revision:1},
    summary:{cells:20},exclusions:{digest:'saved',version:1},
    zarr_save:{state:'saved',shapes_key:'saved_cells',cells:20,result_revision:1,exclusions_digest:'saved'},...extra};
  const sample={id:'s',kind:'zarr',name:'Sample',tools:['proseg','stain'],extent:[0,0,1000,1000],transcripts:500,
    shapes:[],images:[],prior_options:{cells:[{kind:'shapes',key:'saved_cells',label:'Saved cells'}],nuclei:[{kind:'shapes',key:'nuclei',label:'Nuclei'}]},
    suggested_prior:{cells:{kind:'shapes',key:'saved_cells'},nuclei:{kind:'shapes',key:'nuclei'}}};
  const p=page({runOverride:saved,apiOverride:{health:async()=>({proseg:true,one_run:true}),sample:async()=>sample,frames:async()=>({frames:[]})}});
  await settle(30);return {p,saved,sample};
}
test('Proceed checks the layer before leaving and opens an editable, ready cells-only setup',async()=>{
  const {p,saved}=await proceedPage();const button=p.$('proceed');assert.ok(button);assert.equal(button.className,'primary');
  const pressed=button.click();await settle();assert.deepEqual(p.calls.leaves,[['r1','add']]);
  assert.equal(p.main().state.mode,'run');assert.equal(p.$('leave-popup').open,false,'current saved run needs no question');
  p.socket().message({ev:'gone',run:'r1',reason:'left'});await pressed;await settle(20);
  assert.equal(p.main().state.mode,'pick');assert.equal(p.$('run-kind').value,'proseg');
  assert.deepEqual(plain(p.main().state.region),saved.region);
  assert.deepEqual(plain(p.main().state.prior),{cells:{kind:'shapes',key:'saved_cells'},nuclei:null});
  assert.equal(vm.runInContext('pickerProblems().length',p.context),0);assert.equal(p.calls.optionsSet,null,'stain settings never become proseg settings');
  assert.equal(p.calls.fits.at(-1)[1],true,'camera glides to the filled region');
  p.$('prior-choice').value='0';p.$('prior-choice').onchange();assert.equal(p.main().state.prior.nuclei.key,'nuclei','prior stays editable');
});
test('Proceed in a notebook\'s session: cells added in memory only (sopa\'s auto-save off) are never offered; the card says why',async()=>{
  const {p}=await proceedPage({zarr_save:{state:'saved',shapes_key:'saved_cells',cells:20,result_revision:1,exclusions_digest:'saved',
    written_to:null,view_written:null,added:true}});
  p.socket().message({ev:'session',...sessionOf({},{writes:'memory_autosave_off'})});await settle(20);
  assert.equal(p.$('proceed'),null,'no Proceed that could never find its cells');
  assert.ok(p.$('results').textContent.includes(proceed.PROCEED_TEXT.memoryOnly));
  // Written into the zarr the app reads: offered.
  const {p:written}=await proceedPage({zarr_save:{state:'saved',shapes_key:'saved_cells',cells:20,result_revision:1,exclusions_digest:'saved',
    written_to:'/data/sample-1.zarr',view_written:null,added:true}});
  written.socket().message({ev:'session',...sessionOf()});await settle(20);
  assert.equal(written.$('proceed').className,'primary');
});
test('Proceed carries the covered area and clears its note on the first region edit',async()=>{
  const covered={region:{kind:'box',box:[10,20,70,80]},done:2,of:4,why:'tiles'};
  const {p}=await proceedPage({progress:{done:2,total:4,resolved:true,complete:false,revision:1},
    zarr_save:{state:'saved',cells:20,shapes_key:'saved_cells',result_revision:1,exclusions_digest:'saved',covered}});
  const press=p.$('proceed').click();await settle();p.socket().message({ev:'gone',run:'r1',reason:'left'});await press;await settle();
  assert.deepEqual(plain(p.main().state.region),covered.region);assert.match(p.$('picker-notes').textContent,/Limited to the part/);
  vm.runInContext('setRegionMode("full")',p.context);assert.doesNotMatch(p.$('picker-notes').textContent,/Limited to the part/);
});
test('Proceed never leaves stale cells, and image-only results explain the absent button',async()=>{
  const {p}=await proceedPage({exclusions:{digest:'new',version:2}});assert.equal(p.$('proceed').className,'');
  await p.$('proceed').click();assert.equal(p.calls.leaves,undefined);assert.equal(p.main().state.mode,'run');
  const {p:images}=await proceedPage({sample_tools:['stain']});assert.equal(images.$('proceed'),null);
  assert.match(images.$('results').textContent,/no transcripts for proseg/);
});
test('saving in the dialog delays a leave until saved; Stay prevents the later deletion',async()=>{
  const p=page();await settle(20);const state=p.main().state;
  state.run={...state.run,state:'done',zarr_save:{state:'saving',message:'Writing…'}};
  const leaving=vm.runInContext('leave(() => enterPick(null), "new")',p.context);await settle();
  assert.equal(p.$('leave-progress').textContent,'Writing…');assert.equal(p.calls.leaves,undefined);
  await p.$('leave-stay').click();await leaving;
  p.socket().message({ev:'run',run:{...state.run,zarr_save:{state:'saved'}}});await settle();
  assert.equal(p.calls.leaves,undefined);assert.equal(state.mode,'run');
});
test('a run left with a cell selected reaches the picker: the cell card never reads the run already let go',async()=>{
  // The leave clears the run before the picker resets the mixing check, which refreshes the selected cell's card.
  const p=page();await settle(20);p.select(6);await settle(20);
  const next=p.$('new-run').click();await settle();
  if(p.$('leave-popup').open)await p.$('leave-discard').click();
  p.socket().message({ev:'gone',run:'r1',reason:'left'});await next;await settle(30);
  assert.equal(p.main().state.mode,'pick');assert.equal(p.main().state.sample.id,'s');assert.equal(p.$('cell-card').hidden,true);
});
test('a gone event in another window returns to the sample and says what happened',async()=>{
  const p=page();await settle(20);p.socket().message({ev:'gone',run:'r1',reason:'left'});await settle(30);
  assert.equal(p.main().state.mode,'pick');assert.equal(p.main().state.sample.id,'s');
  assert.equal(p.$('notice').textContent,'This run was closed in another window.');
});
test('one-run boot opens finished unsaved results and follows a run created in another window',async()=>{
  const p=page({runOverride:{state:'done'},apiOverride:{health:async()=>({one_run:true,proseg:true})}});await settle(30);
  assert.equal(p.main().state.mode,'run');assert.match(p.$('results').textContent,/Not saved to Sample yet/);
  assert.doesNotMatch(p.$('results').textContent,/Results saved|proseg-output|Copy/);
  const q=page({apiOverride:{health:async()=>({one_run:true,proseg:true}),runs:async()=>[]}});await settle(30);
  assert.equal(q.main().state.mode,'empty');q.socket().message({ev:'runs',runs:[{id:'r1'}]});await settle(30);
  assert.equal(q.main().state.run.id,'r1');assert.match(q.$('notice').textContent,/started in another window/);
});


test('this window\'s own run is never "started in another window": its push before Start\'s answer, while Start opens it, or while the page starts up', async () => {
  const fresh = { id: 'r1', sample: 's', sample_name: 'Sample', sample_kind: 'zarr', state: 'preparing', box: [0, 0, 4, 4], info: null,
    frames: 0, ack: null, options: {} };
  // The engine pushes the runs on every change (the new run's first message), often before Start has it open.
  for (const gated of ['createRun', 'run']) {
    let gate = null; const asked = [];
    const p = page({ apiOverride: { health: async () => ({ one_run: true, proseg: true }), runs: async () => [],
      createRun: async () => { if (gated === 'createRun') await gate.promise; return { ...fresh }; },
      run: async id => { asked.push(id); if (gated === 'run') await gate?.promise; return { ...fresh }; }, frames: async () => ({ frames: [] }) } });
    await settle(30);
    await vm.runInContext('enterPick("s")', p.context); await settle(30);
    gate = deferred();
    const press = p.$('start').click(); await settle(5);
    p.socket().message({ ev: 'runs', runs: [{ ...fresh, message: "Reading the region's transcripts…" }] }); await settle(5);
    gate.resolve(); await press; await settle(30);
    assert.equal(p.main().state.run?.id, 'r1', gated); assert.equal(p.main().state.mode, 'run');
    assert.notEqual(p.$('notice').textContent, p.context.SHELL_TEXT.startedElsewhere, `${gated}: no notice`);
    assert.deepEqual(asked, ['r1'], `${gated}: opened once`);
  }
  // Starting up: the run the page found and opens itself, pushed before it does.
  let schema = null; const asked = [];
  const p = page({ apiOverride: { health: async () => ({ one_run: true, proseg: true }), options: () => (schema = deferred()).promise,
    run: async id => { asked.push(id); return { ...fresh }; }, frames: async () => ({ frames: [] }) } });
  await settle(10); assert.ok(schema, 'the page waits for its options');
  p.socket().message({ ev: 'runs', runs: [{ ...fresh }] }); await settle(10);
  schema.resolve([]); await settle(30);
  assert.equal(p.main().state.run?.id, 'r1'); assert.deepEqual(asked, ['r1'], 'opened once');
  assert.notEqual(p.$('notice').textContent, p.context.SHELL_TEXT.startedElsewhere);
});
test('Open sample clears a discarded run before showing its browser',async()=>{
  const p=page();await settle(20);const pressed=p.$('open-sample').click();await discardRun(p);await pressed;
  assert.equal(p.calls.browserOpened,1);assert.equal(p.main().state.run,null);assert.equal(p.main().state.mode,'pick');
});
test('gone before the leave response runs the navigation once; the late response cannot reopen the run',async()=>{
  const p=page();await settle(20);const gate=deferred();p.context.api.leave=()=>gate.promise;
  const home=p.$('home').click();await settle();const save=p.$('leave-save').click();await settle();
  p.socket().message({ev:'gone',run:'r1',reason:'left'});await home;
  assert.equal(p.main().state.mode,'empty');assert.equal(p.$('leave-popup').open,false);
  gate.resolve({id:'r1',state:'done',zarr_save:{state:'saved'}});await save;
  assert.equal(p.main().state.run,null);assert.equal(p.main().state.mode,'empty');
});
test('Stay during an add cancels leaving in the engine and retains the run when saving completes',async()=>{
  const p=page();await settle(20);const next=p.$('new-run').click();await settle();
  await p.$('leave-save').click();assert.equal(p.$('leave-popup').open,true);
  await p.$('leave-stay').click();await next;assert.deepEqual(p.calls.leaves,[['r1','add'],['r1','stay']]);
  p.socket().message({ev:'run',run:{...p.main().state.run,state:'done',zarr_save:{state:'saved',shapes_key:'cells',table_key:'counts'}}});
  await settle();assert.equal(p.main().state.mode,'run');assert.equal(p.$('leave-popup').open,false);
  assert.match(p.$('results').textContent,/Saved as cells \+ counts/);
});
// A leave the engine could not carry out: Runs._leave keeps the run and pushes it with no leave under way and the
// failure (`leave_error`, its time to the millisecond: each its own); the page never waits for that leave again.
let failures = 0;
const leaveFailed = (p, answer, message = 'The run has not stopped yet.') => p.socket().message({ ev: 'run',
  run: { ...p.main().state.run, leaving: null, leave_error: { answer, message, at: new Date(Date.UTC(2026, 9, 23, 12) + ++failures).toISOString() } } });
const WAYS_OUT = [['the brand', 'home'], ['New run', 'new-run'], ['Open sample', 'open-sample'], ['Close', 'close-app']];
/** What `promise` gave, or 'still waiting' when it has not settled after `n` turns of the event loop. */
const within = (promise, n = 10) => Promise.race([promise, settle(n).then(() => 'still waiting')]);
test('a leave the engine could not carry out, without the prompt: the notice says why, and the brand, New run, Open sample and Close each leave again', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = page({ runOverride: { state: 'stopped', ack: { state: 'stopped' } }, apiOverride: { health: async () => ({ proseg: true, one_run: true }) } });
  await settle(20);
  const { state } = p.main();
  for (const [n, [way, id]] of [...WAYS_OUT, ['the brand again', 'home']].entries()) {
    const left = p.$(id).click(); await settle();
    assert.equal(p.$('leave-popup').open, false, `${way}: nothing to lose, so nothing asked`);
    assert.equal(p.calls.leaves.length, n + 1, `${way}: a leave of its own, never the failed one's wait`);
    assert.deepEqual(p.calls.leaves.at(-1), ['r1', 'discard']);
    leaveFailed(p, 'discard'); assert.equal(await within(left), false, `${way}: the press is over`);
    assert.equal(state.mode, 'run'); assert.equal(p.$('notice').hidden, false);
    assert.equal(p.$('notice').textContent, "Couldn't discard: The run has not stopped yet.");
  }
  // One the engine carries out: the page goes on.
  const home = p.$('home').click(); await settle(); p.socket().message({ ev: 'gone', run: 'r1', reason: 'left' }); await home; await settle(10);
  assert.equal(state.mode, 'empty'); assert.equal(state.run, null);
});
test('two leave failures within the same millisecond are told apart by their ids: the second press ends too', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = page({ runOverride: { state: 'stopped', ack: { state: 'stopped' } }, apiOverride: { health: async () => ({ proseg: true, one_run: true }) } });
  await settle(20);
  const at = '2026-10-23T12:00:00.000Z';
  for (const id of ['first', 'second']) {
    const left = p.$('home').click(); await settle();
    p.socket().message({ ev: 'run', run: { ...p.main().state.run, leaving: null,
      leave_error: { answer: 'discard', message: 'The run has not stopped yet.', at, id } } });
    assert.equal(await within(left), false, `the ${id} failure ends its press`);
  }
  assert.equal(p.calls.leaves.length, 2);
});
test('a leave the engine could not carry out, with the prompt: it says why and asks again, its buttons work, and each way out asks again', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = page();          // a run still sampling: leaving it asks
  await settle(20);
  const { state } = p.main();
  for (const [way, id] of WAYS_OUT) {
    const left = p.$(id).click(); await settle();
    assert.equal(p.$('leave-popup').open, true, `${way} asks`);
    const sent = p.calls.leaves?.length ?? 0;
    await p.$('leave-discard').click(); await settle();
    assert.equal(p.$('leave-discard').getAttribute('aria-busy'), 'true', `${way}: discarding`);
    leaveFailed(p, 'discard'); await settle();
    assert.equal(p.$('leave-popup').open, true, `${way}: the prompt stays`);
    assert.equal(p.$('leave-error').hidden, false); assert.equal(p.$('leave-error').textContent, "Couldn't discard: The run has not stopped yet.");
    assert.equal(p.$('leave-discard').getAttribute('aria-busy'), 'false'); assert.equal(p.$('leave-progress').hidden, true);
    // Discard again goes to the engine (and fails again); Stay ends the leave.
    await p.$('leave-discard').click(); assert.equal(p.calls.leaves.length, sent + 2, `${way}: Discard again is sent`);
    leaveFailed(p, 'discard'); await settle();
    await p.$('leave-stay').click(); assert.equal(await within(left), false, `${way}: the press is over`);
    assert.equal(p.$('leave-popup').open, false); assert.equal(state.mode, 'run'); assert.deepEqual(p.calls.leaves.at(-1), ['r1', 'stay']);
  }
  const home = p.$('home').click(); await settle(); assert.equal(p.$('leave-popup').open, true);
  await p.$('leave-discard').click(); p.socket().message({ ev: 'gone', run: 'r1', reason: 'left' }); await home; await settle(20);
  assert.equal(state.mode, 'empty'); assert.equal(state.run, null);
});
test('in a notebook\'s session a leave the engine could not carry out never holds Close: it ends the session', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = page({ runOverride: { state: 'stopped', ack: { state: 'stopped' } },
    apiOverride: { health: async () => ({ proseg: true, one_run: true, session: sessionOf() }) } });
  await settle(30);
  const answers = []; p.context.api.closing = async (answer, options = {}) => { answers.push([answer, options.runs ?? [], !!options.end]); return {}; };
  const home = p.$('home').click(); await settle();
  assert.deepEqual(p.calls.leaves, [['r1', 'discard']]);
  leaveFailed(p, 'discard'); assert.equal(await within(home), false);
  assert.equal(p.$('notice').textContent, "Couldn't discard: The run has not stopped yet.");
  await within(p.$('close-app').click()); await settle(10);
  assert.deepEqual(plain(answers), [['discard', [], true]], 'Close ends the session');
});
test('a snapshot read before this window\'s leave was taken, arriving after its answer, never ends it with an earlier leave\'s failure', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const earlier = { answer: 'discard', message: 'The run has not stopped yet.', at: '2026-10-23T11:00:00.000+00:00' };
  const p = page({ runOverride: { state: 'stopped', ack: { state: 'stopped' }, leave_error: earlier },
    apiOverride: { health: async () => ({ proseg: true, one_run: true }) } });
  await settle(20);
  const { state } = p.main();
  const failedNotice = () => !p.$('notice').hidden && /Couldn't|another window/.test(p.$('notice').textContent);
  for (const ending of ['a failure of its own', 'the engine carries it out']) {
    const stale = { ...state.run };            // what a runs poll read before the leave was taken
    p.context.api.leave = async (id, answer) => { (p.calls.leaves ??= []).push([id, answer]);
      return { ...stale, leave_error: null, leaving: { answer, stage: 'finishing', message: 'Discarding the run…' } }; };
    const left = p.$('home').click(); await settle();
    assert.deepEqual(p.calls.leaves.at(-1), ['r1', 'discard'], ending);
    p.socket().message({ ev: 'run', run: stale }); await settle();
    assert.equal(await within(left), 'still waiting', `${ending}: the earlier failure is not this leave's`);
    assert.equal(failedNotice(), false, ending);
    if (ending === 'a failure of its own') {
      leaveFailed(p, 'discard'); assert.equal(await within(left), false, 'its own failure ends it');
      assert.equal(p.$('notice').textContent, "Couldn't discard: The run has not stopped yet.");
      p.$('notice').hidden = true;
    } else {
      p.socket().message({ ev: 'gone', run: 'r1', reason: 'left' }); assert.notEqual(await within(left, 20), 'still waiting');
      await settle(10);
      assert.equal(state.mode, 'empty'); assert.equal(state.run, null); assert.equal(failedNotice(), false, 'never "closed in another window"');
    }
  }
});
test('Proceed whose leave the engine could not carry out is free again at once and says why; every way out and Proceed leave again, also after a prompt for a leave under way', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { p } = await proceedPage(), { state } = p.main();
  let pressed = p.$('proceed').click(); await settle();
  assert.deepEqual(p.calls.leaves, [['r1', 'add']]); assert.equal(p.$('proceed').getAttribute('aria-busy'), 'true');
  leaveFailed(p, 'add'); assert.equal(await within(pressed), undefined, 'the press is over'); await settle();
  assert.equal(p.$('proceed').getAttribute('aria-busy'), null); assert.equal(p.$('proceed').textContent, 'Proceed to proseg');
  assert.equal(p.$('notice').textContent, "Couldn't leave this run: The run has not stopped yet.", 'saved already: only leaving it failed');
  for (const [way, id] of WAYS_OUT) {
    const left = p.$(id).click(); await settle();
    assert.equal(p.$('leave-popup').open, false); assert.deepEqual(p.calls.leaves.at(-1), ['r1', 'add'], way);
    leaveFailed(p, 'add'); assert.equal(await within(left), false, way); assert.equal(state.mode, 'run');
  }
  assert.equal(p.calls.leaves.length, 1 + WAYS_OUT.length);
  // A leave under way from another window: Proceed's prompt says it, then its failure; Stay frees Proceed.
  p.socket().message({ ev: 'run', run: { ...state.run, leaving: { answer: 'discard', stage: 'finishing', message: 'Discarding the run…' } } });
  await settle(10);
  pressed = p.$('proceed').click(); await settle(10);
  assert.equal(p.$('leave-popup').open, true); assert.equal(p.$('leave-progress').textContent, 'Discarding…');
  leaveFailed(p, 'discard'); await settle();
  assert.equal(p.$('leave-error').textContent, "Couldn't discard: The run has not stopped yet.");
  assert.equal(p.$('proceed').getAttribute('aria-busy'), 'true', 'the prompt is still asking');
  await p.$('leave-stay').click(); assert.equal(await within(pressed), undefined, 'Stay ends the press'); await settle();
  assert.equal(p.$('leave-popup').open, false); assert.equal(p.$('proceed').getAttribute('aria-busy'), null);
  // Proceed again: the engine leaves it, and the proseg setup opens.
  pressed = p.$('proceed').click(); await settle(); assert.deepEqual(p.calls.leaves.at(-1), ['r1', 'add']);
  p.socket().message({ ev: 'gone', run: 'r1', reason: 'left' }); await pressed; await settle(20);
  assert.equal(state.mode, 'pick'); assert.equal(p.$('run-kind').value, 'proseg');
});
test('the poll recovers a missed gone push for a finished run',async()=>{
  const p=page({runOverride:{state:'done'},apiOverride:{health:async()=>({one_run:true,proseg:true})}});await settle(30);
  p.context.api.run=async()=>{throw Object.assign(new Error('Unknown run'),{status:404});};
  await p.calls.intervals.find(item=>item.ms===1500).fn();await settle(20);
  assert.equal(p.main().state.mode,'pick');assert.equal(p.$('notice').textContent,'This run was closed in another window.');
});
test('older runs: listed before anything opens, the start notice in the lead; a row saved, the rest discarded together; then the page goes on',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});      // the saved row's moment and glide, told by the test
  const row=(id,extra={})=>({id,kind:'proseg',state:'done',sample:'s',sample_name:`Sample ${id}`,created:'2026-10-14T14:42:00+00:00',region:{kind:'full'},
    prior:{cells:{kind:'shapes',key:'cellpose_boundaries_1'},nuclei:null},options:{},can_add:true,why:'unsaved',...extra});
  let rows=[row('old-a'),row('old-b'),row('old-c',{kind:'stain',method:'cellpose',stain:{model:'cpsam',nucleus:'DAPI'},tiles:{done:3,of:8},progress:{done:3,total:8}})];
  let done=[];
  const posts=[],gate=deferred();
  const p=page({apiOverride:{health:async()=>({one_run:true,proseg:true,older:3,leftovers:{stamp:`older-${Math.random()}`,runs:[{sample:'SMP02'}],older_removed:2}}),
    runs:async()=>[],olderRuns:async()=>({runs:rows.map(r=>({...r})),done:done.map(r=>({...r}))}),
    // The engine (engine/runs.py Runs.older): a row's answer queued, its stage and message while it works; Discard all
    // takes the rows not under way.
    answerOlder:async(id,answer)=>{posts.push([id,answer]);await gate.promise;
      if(id==='old-a')rows=rows.map(r=>r.id===id?{...r,answer,stage:'writing',message:'Saving to the sample…'}:r);
      else rows=rows.filter(r=>r.id==='old-a');
      return {runs:rows.map(r=>({...r})),done:[],count:rows.length};}}});
  await settle(30);
  const prompt=vm.runInContext('olderPrompt',p.context);t.after(()=>prompt.finish());   // its poll stops with it
  assert.equal(prompt.open,true,'the prompt before anything opens');assert.equal(p.main().state.mode,'empty');
  assert.equal(p.$('older-title').textContent,'Save your older runs?');
  assert.match(p.$('older-notice').textContent,/A run on SMP02 that was not saved to the sample was removed when the app started/);
  assert.match(p.$('older-notice').textContent,/2 runs from earlier versions were removed: they were saved already or had nothing to save\./);
  assert.equal(p.$('notice').hidden,true,'no separate start notice behind the prompt');
  const lines=prompt.dialog.all().filter(n=>n.className==='older-line').map(n=>n.textContent);
  assert.match(lines[0],/^Sample old-a · whole sample · Proseg · Finished · \d+ Oct \d\d:\d\d$/);
  assert.match(lines[2],/^Sample old-c · whole sample · Cellpose · 3 of 8 tiles done · /);
  // A row's Save: sent once, its buttons go while the engine works on it.
  const saveA=prompt.dialog.all().find(n=>n.getAttribute?.('data-older-run')==='old-a').all().find(n=>n.getAttribute?.('data-older-answer')==='add');
  const pressed=saveA.click();await settle();assert.deepEqual(posts,[['old-a','add']]);
  gate.resolve();await pressed;await settle();
  assert.match(prompt.dialog.all().find(n=>n.getAttribute?.('data-older-run')==='old-a').textContent,/Saving to the sample…/);
  // Discard all: one request for the rows left to answer; the engine keeps the one being saved.
  await p.$('older-discard-all').click();await settle();assert.deepEqual(posts.at(-1),[null,'discard']);
  assert.deepEqual([...prompt.rows.keys()],['old-a']);
  // The save ends: the row says what it saved a moment and glides out; then the prompt closes and the page goes on (the
  // first screen: no run open).
  rows=[];done=[{id:'old-a',stage:'saved',message:'Saved as proseg_boundaries_2 + proseg_table_2',
    zarr_save:{state:'saved',shapes_key:'proseg_boundaries_2',table_key:'proseg_table_2'}}];
  await vm.runInContext('olderPrompt.poll()',p.context);await settle(10);
  assert.equal(prompt.open,true,'the last row is seen saved');assert.equal(p.main().state.mode,'empty');
  assert.match(prompt.dialog.all().find(n=>n.getAttribute?.('data-older-run')==='old-a').textContent,/Saved as proseg_boundaries_2 \+ proseg_table_2/);
  t.mock.timers.tick(p.context.SAVED_MS);await settle(5);
  for(let i=0;i<8;i++){t.mock.timers.tick(100);await settle(2);}
  assert.equal(prompt.open,false);assert.equal(p.main().state.mode,'empty');
  assert.equal(p.$('tool-choice').hidden,false);
});
test('older runs: Later (or Esc) lets the page go on at once, the engine setting the rows aside; a run open in the engine then opens',async t=>{
  const posts=[];
  const p=page({apiOverride:{health:async()=>({one_run:true,proseg:true,older:{runs:[{id:'old-a',state:'done',sample_name:'Sample A',can_add:true}]}}),
    olderRuns:async()=>{throw Object.assign(new Error('Not Found'),{status:404});},answerOlder:async(...a)=>{posts.push(a);}}});
  await settle(30);
  const prompt=vm.runInContext('olderPrompt',p.context);t.after(()=>prompt.finish());
  assert.equal(prompt.open,true,'an engine without GET /runs/older: the rows health named');assert.equal(p.main().state.run,null);
  assert.equal(p.$('older-later').getAttribute('data-older-answer'),'later');
  prompt.dialog.emit('cancel');await settle(30);
  assert.equal(prompt.open,false);assert.deepEqual(posts,[[null,'later']],'Later: one request, for every row');
  assert.equal(p.main().state.mode,'run','the run the engine holds opens after the prompt');
});
test('leave waits for filters, then uses the settled kept count, with a five-second limit',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const {p}=await proceedPage({zarr_save:null});const gate=deferred();p.main().stainExclusions.whenSettled=()=>gate.promise;
  const next=p.$('new-run').click();await settle();assert.equal(p.$('leave-popup').open,false);
  const state=p.main().state;state.run={...state.run,summary_model:{cells:20},summary:{cells:0}};
  gate.resolve();await settle();assert.equal(p.$('leave-save').hidden,true);await p.$('leave-stay').click();await next;
  p.main().stainExclusions.whenSettled=()=>new Promise(()=>{});
  const late=p.$('new-run').click();await settle();t.mock.timers.tick(5000);await late;
  assert.equal(p.calls.leaves,undefined);assert.equal(state.mode,'run');assert.match(p.$('notice').textContent,/Wait for the cell changes/);
});
test('a removal panel that could not read its areas never keeps the window on the run: the prompt still asks',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const {p}=await proceedPage({zarr_save:null});
  p.main().stainExclusions.whenSettled=async()=>{throw new Error('The removed areas could not be loaded. Reopen the run to try again.');};
  const home=p.$('home').click();await settle();
  assert.equal(p.$('leave-popup').open,true,'the brand leaves as ever: the prompt asks to save or discard');
  assert.doesNotMatch(p.$('notice').textContent,/could not be loaded/);
  await p.$('leave-stay').click();await home;assert.equal(p.main().state.mode,'run');
});
test('an old poll cannot make changed filters look saved in the leave prompt',async()=>{
  const {p,saved}=await proceedPage({exclusions:{version:2,digest:'changed'},zarr_save:{state:'saving'}});
  const next=p.$('new-run').click();await settle();
  p.socket().message({ev:'run',run:{...saved,exclusions:{version:1,digest:'saved'},
    zarr_save:{state:'saved',shapes_key:'saved_cells',cells:20,result_revision:1,exclusions_digest:'saved'}}});await settle();
  assert.equal(p.calls.leaves,undefined);assert.equal(p.main().state.run.exclusions.digest,'changed');
  assert.match(p.$('leave-run').textContent,/changed since/);await p.$('leave-stay').click();await next;
});
test('Proceed waits for filters and rechecks the saved key after gone',async()=>{
  const {p,saved,sample}=await proceedPage();const gate=deferred();p.main().stainExclusions.whenSettled=()=>gate.promise;
  const pressed=p.$('proceed').click();await settle();assert.equal(p.calls.leaves,undefined);
  gate.resolve();await settle();assert.deepEqual(p.calls.leaves,[['r1','add']]);
  const changed={...saved,zarr_save:{...saved.zarr_save,shapes_key:'new_saved_cells'}};
  p.context.api.sample=async()=>({...sample,prior_options:{...sample.prior_options,cells:[{kind:'shapes',key:'new_saved_cells',label:'New saved cells'}]}});
  p.socket().message({ev:'run',run:changed});p.socket().message({ev:'gone',run:'r1',reason:'left'});await pressed;await settle();
  assert.deepEqual(plain(p.main().state.prior),{cells:{kind:'shapes',key:'new_saved_cells'},nuclei:null});
});
test('Proceed stays on the stain run if its saved key is absent after the retry',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});const {p,sample}=await proceedPage();
  let reads=0;p.context.api.sample=async()=>{reads++;return {...sample,prior_options:{cells:[],nuclei:[]}};};
  const pressed=p.$('proceed').click();await settle();assert.equal(reads,1);t.mock.timers.tick(1000);await pressed;
  assert.equal(reads,2);assert.equal(p.calls.leaves,undefined);assert.equal(p.main().state.mode,'run');
  assert.match(p.$('results').textContent,/saved cells are not offered yet/);
});

test('range integration forwards classes and preserves the full exclusion state against an older read',async()=>{
  const {p,saved}=await proceedPage(),{priors,state}=p.main(),o=p.calls.exclusions;
  assert.equal(o.ranges.size,p.$('stain-exclude-size-row'));assert.equal(o.ranges.brightness,p.$('stain-exclude-bright-row'));
  assert.equal(o.priors,priors);const classes={gen:2,set:'vectors-1',array:Uint8Array.of(0,1,0)};
  o.classes(classes);assert.equal(priors.classes,classes);assert.equal(p.main().stage.dirty,true);
  p.context.api.run=async()=>saved;
  await o.refresh('r1',{version:4,areas_version:2,digest:'changed',count:0,kept:12,cells:20,brightness:{DAPI:{min:10,max:80}}});
  assert.equal(state.run.exclusions.digest,'changed');assert.equal(state.run.exclusions.areas_version,2);
  assert.equal(state.run.summary.cells,12);assert.equal(state.run.exclusions.brightness.DAPI.max,80);
});


test('a browser without WebGL2 says the window cannot draw, and what to do, and starts nothing else', () => {
  // A notebook's window on a remote display where Chrome blocked WebGL2 and the page stayed empty: the app's own Chrome
  // window (window=chrome in its opening address) is told which of its settings to change.
  const asked = [], engine = Object.fromEntries(['health', 'samples', 'runs', 'options'].map(name => [name, async () => { asked.push(name); return []; }]));
  const p = page({ stageError: new Error('WebGL2 is not available in this browser.'), search: '?window=chrome', apiOverride: engine });
  assert.match(String(p.thrown?.message), /WebGL2/, 'the error goes on, for the page\'s log');
  const box = p.$('no-webgl');
  assert.equal(box.hidden, false);
  assert.equal(p.$('no-webgl-title').textContent, p.context.SHELL_TEXT.noWebglTitle);
  assert.equal(p.$('no-webgl-why').textContent, p.context.SHELL_TEXT.noWebglWhy);
  assert.match(p.$('no-webgl-help').textContent, /--ignore-gpu-blocklist/);
  assert.equal(p.$('empty').hidden, true, 'nothing else of the page started');
  // Any other browser (the default browser, a printed link) has no chrome_flags: its own words.
  const other = page({ stageError: new Error('WebGL2 is not available in this browser.'), apiOverride: engine });
  assert.equal(other.$('no-webgl-help').textContent, SHELL_TEXT.noWebglHelpBrowser);
  assert.doesNotMatch(other.$('no-webgl-help').textContent, /chrome_flags|--ignore-gpu-blocklist/);
  // The app's window after the page rewrote its address: the tab remembers it is the app's window.
  const store = new Map([[WINDOW_STORE, 'chrome']]), tab = { getItem: key => store.get(key) ?? null };
  const rewritten = page({ stageError: new Error('WebGL2 is not available in this browser.'), search: '?run=r1', tabStorage: tab, apiOverride: engine });
  assert.equal(rewritten.$('no-webgl-help').textContent, SHELL_TEXT.noWebglHelp);
  assert.deepEqual(asked, [], 'none of them asked the engine anything');
});

// ---- signing in (page-key.mjs), the page going, and Close in a tab the page cannot close ---------------------------
/** Answers that record that the engine was asked: a page that is not signed in asks it nothing. */
const recorded = asked => Object.fromEntries(['health', 'samples', 'runs', 'options', 'optionsGuide', 'run', 'frames', 'stainModels', 'olderRuns']
  .map(name => [name, async () => { asked.push(name); return []; }]));
const fire = (p, type, event) => { for (const fn of p.context.window.listeners.get(type) || []) fn(event); };

test('a tab without a key asks the engine nothing and says it is not signed in, with nothing to press', async () => {
  const asked = [], p = page({ key: null, keyReason: 'none', apiOverride: recorded(asked), hook: true });
  await settle(20);
  assert.deepEqual([p.$('engine-down').hidden, p.$('engine-down-title').textContent, p.$('engine-down-why').textContent, p.$('engine-reload').hidden],
    [false, SHELL_TEXT.signedOutTitle, SHELL_TEXT.signedOut, true], 'Reload would sign nothing in: hidden, never greyed out');
  assert.deepEqual([p.main().state.mode, p.$('tool-choice').hidden, p.snapshot().signedOut], ['empty', true, 'none']);
  assert.deepEqual(asked, [], 'nothing asked of the engine');
  assert.deepEqual(p.calls.socket, ['stop']);
  // A link opened again, or too late: the link's words.
  const used = page({ key: null, keyReason: 'expired', apiOverride: recorded(asked) });
  await settle(20);
  assert.deepEqual([used.$('engine-down-title').textContent, used.$('engine-down-why').textContent], [SHELL_TEXT.linkUsedTitle, SHELL_TEXT.linkUsed]);
  // A notebook's session's tab, its link naming the session or the tab remembering it: from the notebook.
  const session = page({ key: null, keyReason: 'refused', search: '?api=1f3a9c2e', apiOverride: recorded(asked) });
  await settle(20);
  assert.deepEqual([session.$('engine-down-title').textContent, session.$('engine-down-why').textContent], [SESSION_TEXT.signedOutTitle, SESSION_TEXT.signedOut]);
  const tab = { getItem: key => key === p.context.MARK ? '1f3a9c2e' : null };
  const remembered = page({ key: null, keyReason: 'expired', tabStorage: tab, apiOverride: recorded(asked) });
  await settle(20);
  assert.deepEqual([remembered.$('engine-down-title').textContent, remembered.$('engine-down-why').textContent], [SESSION_TEXT.linkUsedTitle, SESSION_TEXT.linkUsed]);
  assert.deepEqual(asked, []);
  // The engine could not be reached to sign in: said as an engine that does not answer, with Reload.
  const away = page({ key: null, keyReason: 'unreachable', apiOverride: recorded(asked) });
  await settle(20);
  assert.deepEqual([away.$('engine-down-title').textContent, away.$('engine-down-why').textContent, away.$('engine-reload').hidden],
    [SHELL_TEXT.engineDown, NOTE_TEXT.noAnswer, false]);
  assert.equal(away.main().state.signedOut, undefined);
  for (const text of [SHELL_TEXT.signedOutTitle, SHELL_TEXT.signedOut, SHELL_TEXT.linkUsedTitle, SHELL_TEXT.linkUsed, SESSION_TEXT.signedOut,
    SESSION_TEXT.linkUsed]) assert.doesNotMatch(text, /sdata|cookie|token|key/i, text);
});

test('Close on a page not signed in closes its window if it may, and otherwise leaves the page saying how to open the app again', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = page({ key: null, keyReason: 'none' });
  await settle(20);
  const win = p.context.window;
  win.closes = 0; win.closed = false; win.close = () => { win.closes++; };
  p.$('close-app').click(); await settle(); t.mock.timers.tick(CLOSE_CHECK_MS); await settle();
  assert.deepEqual([win.closes, p.calls.socket], [1, ['stop']], 'no goodbye: it closed nothing of the app');
  assert.deepEqual([p.$('engine-down-title').textContent, p.main().state.closed], [SHELL_TEXT.signedOutTitle, undefined]);
});

test('a key the engine no longer takes, mid-run: the page says it is signed out in place of the run, once, and nothing over it', async () => {
  const p = page({ hook: true });
  await settle(20);
  const { state } = p.main();
  assert.equal(state.mode, 'run');
  fire(p, 'sopa-signed-out', { detail: 'refused' });
  await settle(5);
  assert.deepEqual([state.signedOut, state.mode, state.run, p.$('engine-down').hidden, p.$('engine-down-title').textContent, p.$('engine-reload').hidden],
    ['refused', 'empty', null, false, SHELL_TEXT.signedOutTitle, true]);
  assert.deepEqual([p.calls.socket, p.snapshot().signedOut], [['stop'], 'refused']);
  p.$('notice').hidden = true;
  vm.runInContext("notice('Request failed (500). Please try again.')", p.context);
  assert.equal(p.$('notice').hidden, true, 'nothing is said over it');
  fire(p, 'sopa-signed-out', { detail: 'refused' });
  assert.deepEqual(p.calls.socket, ['stop'], 'once');
});

test('a boot refused for its key (the engine started again since) is signed out, not an engine that cannot be reached', async () => {
  const refused = async () => { throw Object.assign(new Error('A SoPA companion page key or bearer token is required'), { status: 401 }); };
  const p = page({ apiOverride: { health: refused } });
  await settle(20);
  assert.deepEqual([p.main().state.signedOut, p.$('engine-down-title').textContent, p.$('engine-down-why').textContent],
    ['refused', SHELL_TEXT.signedOutTitle, SHELL_TEXT.signedOut]);
});

test('the page going says goodbye on its socket; a page the browser brings back connects again, a closed one does not', async () => {
  const p = page();
  await settle(20);
  p.context.api.closing = async () => ({});
  fire(p, 'pagehide', { persisted: true });
  assert.deepEqual(p.calls.socket, ['goodbye']);
  fire(p, 'pageshow', { persisted: false });
  assert.deepEqual(p.calls.socket, ['goodbye'], 'a page shown for the first time is not one coming back');
  fire(p, 'pageshow', { persisted: true });
  assert.deepEqual(p.calls.socket, ['goodbye', 'back']);
});

test('Close in a tab the page cannot close: it says goodbye, then that the app has closed and the tab can be closed', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = page({ hook: true, apiOverride: { runs: async () => [] } });
  await settle(20);
  const win = p.context.window;
  win.closes = 0; win.closed = false; win.close = () => { win.closes++; };      // a tab the user opened: it stays open
  p.$('close-app').click(); await settle();
  assert.deepEqual([win.closes, p.calls.socket], [1, ['goodbye']], 'the engine hears the page go before the window closes');
  t.mock.timers.tick(CLOSE_CHECK_MS); await settle();
  assert.deepEqual([p.$('engine-down').hidden, p.$('engine-down-title').textContent, p.$('engine-down-why').textContent, p.$('engine-reload').hidden],
    [false, SHELL_TEXT.closedTitle, SHELL_TEXT.closedTab, true]);
  assert.deepEqual([p.calls.socket, p.snapshot().closed, p.main().state.mode], [['goodbye', 'stop'], true, 'empty']);
  assert.equal(p.$('notice').hidden, true, 'no word that the window cannot close itself');
  // The engine is going: a request that fails now is no news on the closed screen.
  fire(p, 'fetch-notice', { detail: 'Connection failed. Please try again.' });
  assert.equal(p.$('notice').hidden, true, 'nothing is said over the closed screen');
  fire(p, 'pageshow', { persisted: true });
  assert.deepEqual(p.calls.socket, ['goodbye', 'stop'], 'brought back after it closed, it connects nothing again');
  // A window that does close needs no words.
  const q = page({ apiOverride: { runs: async () => [] } });
  await settle(20);
  const shut = q.context.window; shut.closed = false; shut.close = () => { shut.closed = true; };
  q.$('close-app').click(); await settle(); t.mock.timers.tick(CLOSE_CHECK_MS); await settle();
  assert.equal(q.$('engine-down').hidden, true);
});

test('Close with the app still open in another tab says this tab is closed, not the app; the engine asked as Close is pressed', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const asked = [], pages = { count: 2 };
  const health = async signal => { asked.push(signal); return { proseg: true, presence: { pages: pages.count, state: 'here' } }; };
  const closeIn = async (p, open = true) => {
    await settle(20);
    const win = p.context.window;
    win.closed = false; win.close = () => {};                                  // a tab the user opened: it stays open
    p.calls.socketOpen = open;
    const before = asked.length;
    p.$('close-app').click(); await settle();
    assert.equal(asked.length, before + 1, 'asked once, before the goodbye, with a time limit');
    assert.ok(asked.at(-1)?.aborted === false);
    t.mock.timers.tick(CLOSE_CHECK_MS); await settle();
    return [p.$('engine-down-title').textContent, p.$('engine-down-why').textContent, p.$('engine-reload').hidden];
  };
  // Two pages open, this tab's socket one of them: another tab keeps the app open.
  const p = page({ hook: true, apiOverride: { runs: async () => [], health } });
  assert.deepEqual(await closeIn(p), [SHELL_TEXT.closedHereTitle, SHELL_TEXT.closedHere, true]);
  assert.deepEqual([p.calls.socket, p.snapshot().closed], [['goodbye', 'stop'], true]);
  // Its own socket the only one: the app has closed. Its socket not open (connecting again): the one open is another's.
  pages.count = 1;
  assert.deepEqual(await closeIn(page({ apiOverride: { runs: async () => [], health } })), [SHELL_TEXT.closedTitle, SHELL_TEXT.closedTab, true]);
  assert.deepEqual(await closeIn(page({ apiOverride: { runs: async () => [], health } }), false),
    [SHELL_TEXT.closedHereTitle, SHELL_TEXT.closedHere, true]);
  // The engine not answering (going already): the app has closed.
  let asks = 0;
  const failing = async signal => { asked.push(signal); if (++asks > 1) throw new TypeError('Failed to fetch'); return { proseg: true }; };
  assert.deepEqual(await closeIn(page({ apiOverride: { runs: async () => [], health: failing } })),
    [SHELL_TEXT.closedTitle, SHELL_TEXT.closedTab, true]);
});

test('a run the engine paused while no window showed it says why once, as it opens and as it is pushed, for each pause', async () => {
  const away = '2026-10-25T10:00:00+00:00';
  const p = page({ runOverride: { state: 'paused', paused: true, paused_away: away } });
  await settle(20);
  const { state } = p.main();
  assert.deepEqual([state.run.id, p.$('notice').hidden, p.$('notice').textContent], ['r1', false, SHELL_TEXT.awayPaused]);
  p.$('notice').textContent = ''; p.$('notice').hidden = true;
  p.socket().message({ ev: 'run', run: { ...state.run, paused_away: away } }); await settle();
  assert.equal(p.$('notice').hidden, true, 'the same pause is said once');
  p.socket().message({ ev: 'run', run: { ...state.run, id: 'other', paused_away: away } }); await settle();
  assert.equal(p.$('notice').hidden, true, 'never for a run not on screen');
  p.socket().message({ ev: 'run', run: { ...state.run, paused_away: '2026-10-25T11:00:00+00:00' } }); await settle();
  assert.equal(p.$('notice').textContent, SHELL_TEXT.awayPaused, 'another pause is said again');
  p.$('notice').textContent = ''; p.$('notice').hidden = true;
  p.socket().message({ ev: 'run', run: { ...state.run, paused: false, state: 'running', paused_away: undefined } }); await settle();
  assert.equal(p.$('notice').hidden, true, 'resumed: nothing to say');
  const q = page({ runOverride: { state: 'paused', paused: true } });   // paused by its Pause button: nothing said
  await settle(20);
  assert.notEqual(q.$('notice').textContent, SHELL_TEXT.awayPaused);
});

// ---- the range choosers through main.js: outlines hidden live, one POST on release, Undo glides them back -----------
import { toPolygonSet as WIRE_toPolygonSet } from '../src/polygons.mjs';
import { layerPath as WIRE_layerPath } from '../src/prior-paths.mjs';
test('WIRE: ranges hide outlines live through main.js; one POST on release; Undo brings them back; no vectors refetch', async t => {
  // The engine's own bytes (engine/vectors.py encode_cells, pinned by tests/test_vectors.py): three cells of 5, 20 and
  // 80 µm², DAPI 1, unknown and 100.
  const fixture=name=>{const b=readFileSync(new URL(`./fixtures/${name}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
  const vectorFetches=[];
  const polys=[1,2,3].map(id=>({id,poly:[[id*20,10],[id*20+8,10],[id*20+8,18],[id*20,18]]}));
  t.mock.method(globalThis,'fetch',async url=>{
    if(/\/stain\/vectors\?/.test(String(url))){vectorFetches.push(String(url));return {ok:true,arrayBuffer:async()=>encodeVectors({cells:polys},{set:'resolved',areas_version:0,revision:2})};}
    throw Error('unexpected fetch '+url);
  });
  const base={kind:'stain',state:'paused',paused:true,ack:{state:'paused'},info:null,frames:0,
    stain:{model:'cpsam',image:'image',nucleus:'DAPI',tissue_only:true},box:[0,0,100,100],
    exclusions:{version:0,areas_version:0},progress:{done:1,total:1,revision:2,resolved:true},summary:{cells:3}};
  const p=page({realStain:true,realRoundtrip:true,runOverride:base});
  let engine={version:0,areas_version:0,areas:[],size:null,brightness:{},cells:3,kept:3,removed:0,by_size:0,by_brightness:0,in_areas:0};
  const posts=[],cellReads=[];
  Object.assign(p.context.api,{
    stainExclusions:async()=>JSON.parse(JSON.stringify(engine)),
    stainCells:async(id,set,channel)=>{cellReads.push([id,set,channel]);return fixture(channel?'stain-cells-dapi.bin':'stain-cells-none.bin');},
    stainBrightness:async()=>({channel:'DAPI',channels:['DAPI'],default:'DAPI',p10:1}),
    stainExcludeSize:async(id,body)=>{posts.push(body);engine={...engine,version:engine.version+1,
      size:body.min===null&&body.max===null?null:{min:body.min,max:body.max}};return JSON.parse(JSON.stringify(engine));},
    run:async()=>({id:'r1',...base,exclusions:{version:engine.version,areas_version:0,count:0}}),
  });
  await settle(60);
  const ex=p.main().stainExclusions,priors=p.main().priors;
  assert.equal(vectorFetches.length,1,'vectors fetched once');
  assert.equal(ex.vectors?.set,'resolved','the vectors reach the panel');
  assert.deepEqual(cellReads.map(r=>r[2]),[null,'DAPI'],'values read: sizes, then the channel in use');
  assert.equal(ex.sizeChooser.axis.count,3,'histogram filled');
  assert.equal(ex.sizeChooser.sliders.max.hidden,false,'handles shown');
  const drawn=()=>{const set=WIRE_toPolygonSet(priors.vectors.cells);const array=priors.classes?.set===priors.vectors.set?priors.classes.array:null;
    return WIRE_layerPath(set,[0,0,100,100],1,class{addPath(){}moveTo(){}lineTo(){}closePath(){}},false,1e6,array).polygons;};
  assert.equal(drawn(),3,'all outlines before a drag');
  const c=ex.sizeChooser,s=c.sliders.max;
  s.emit('pointerdown',{button:0,pointerId:1});s.value=String(c.axis.position(10)*1000);s.emit('input');await settle(5);
  assert.equal(posts.length,0,'nothing posted while held');
  assert.deepEqual([...priors.classes.array],[0,1,1]);assert.equal(drawn(),1,'outlines hidden while held');
  assert.equal(p.$('stain-exclude-total').textContent,'Keeps 1 of 3 cells');
  s.emit('change');await settle(20);
  assert.deepEqual(JSON.parse(JSON.stringify(posts)),[{version:0,min:null,max:10}],'one POST on release');
  assert.equal(ex.undos.length,1,'one Undo step');assert.equal(drawn(),1,'still hidden after release');
  await p.$('stain-exclude-undo').click();await settle(20);
  assert.deepEqual(JSON.parse(JSON.stringify(posts.at(-1))),{version:1,min:null,max:null},'Undo posts the old range');assert.equal(posts.length,2);
  // The handles glide back over 200 ms and the outlines return as they pass them.
  for(let i=0;i<30 && ex.tweens.size;i++){await new Promise(r=>setTimeout(r,20));await settle(2);}
  assert.equal(drawn(),3,'Undo brings the outlines back');
  assert.equal(vectorFetches.length,1,'no vectors refetch');
});
