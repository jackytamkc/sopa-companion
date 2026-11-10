import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import { harness, createHook, install, enabled, withFlag, VERSION } from '../src/harness-hook.mjs';
import { fakeLoop, fakeElement } from './fake-loop.mjs';
import { polygonCount } from '../src/polygons.mjs';

// The development harness's read-only hook: only with ?harness=1, and then counted at the app's own sites.
const SRC = new URL('../src/', import.meta.url);
const dataURL = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
let loads = 0;
/** src/`file` with its harness-hook import bound to `hook` (every other relative import: the real module). */
async function withHook(file, hook) {
  globalThis.__plvTestHook = hook;   // read once, when the stand-in module is evaluated
  const stand = dataURL(`export const harness = globalThis.__plvTestHook; // ${++loads}`);
  const source = (await readFile(new URL(file, SRC), 'utf8')).replace(/from '\.\/([^']+)'/g, (_, name) =>
    `from ${JSON.stringify(name === 'harness-hook.mjs' ? stand : new URL(name, SRC).href)}`);
  return import(dataURL(`${source}\n// ${loads}`));
}
const clock = (t = 0) => ({ t, timeOrigin: 1700000000000.5, now() { return this.t; } });
const counters = hook => hook.snapshot().counters;
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const settle = async (n = 5) => { for (let i = 0; i < n; i++) await new Promise(resolve => setImmediate(resolve)); };
/** A WebGL2 context that accepts every call: constants are distinct numbers, objects fresh tokens, programs have no
 *  active uniforms and every compile and link succeeds. */
function fakeGL() {
  const ids = new Map(), constant = key => { if (!ids.has(key)) ids.set(key, 1000 + ids.size); return ids.get(key); };
  return new Proxy({}, { get: (_, key) => typeof key !== 'string' ? undefined : /^[A-Z0-9_]+$/.test(key) ? constant(key) : (...args) =>
    key === 'getParameter' ? 4096 : key === 'getShaderParameter' || key === 'getProgramParameter' ? (args[1] === constant('ACTIVE_UNIFORMS') ? 0 : true) :
      key === 'getExtension' ? null : key === 'isContextLost' ? false : {} });
}
const plain = value => JSON.parse(JSON.stringify(value));   // objects built in a vm context have that context's prototypes
function page(search, extra = {}) {
  const calls = [];
  const history = { replaceState(...args) { calls.push(['replaceState', ...args]); }, pushState(...args) { calls.push(['pushState', ...args]); } };
  return { calls, win: { location: { search, href: `http://127.0.0.1:8810/${search}` }, history, performance: clock(5), fetch: async () => 'native', ...extra } };
}

test('the flag is exactly harness=1 in the query; URLs the app writes keep it', () => {
  for (const search of ['?harness=1', '?run=r1&harness=1', 'harness=1', '?harness=1&harness=0']) assert.equal(enabled(search), true, search);
  for (const search of ['', '?', '?harness', '?harness=0', '?harness=true', '?run=harness=1', undefined, null]) assert.equal(enabled(search), false, String(search));
  const base = 'http://127.0.0.1:8810/?harness=1';
  assert.equal(withFlag('?run=r%2F1', base), '/?run=r%2F1&harness=1');   // openRun: ?run=<id>
  assert.equal(withFlag('/', base), '/?harness=1');                       // enterPick: location.pathname
  assert.equal(withFlag('/?harness=1&run=a', base), '/?harness=1&run=a', 'already flagged: unchanged');
  assert.equal(withFlag('#top', 'http://h/?run=a'), '/?run=a&harness=1#top');
  assert.equal(withFlag('http://evil.example/x', base), 'http://evil.example/x', 'another origin: history refuses it, unchanged');
});

test('without the flag nothing is installed: no hook, no wrapped fetch or history, no window.__plvHarness', () => {
  assert.equal(harness, null, 'node has no page: the module exports null');
  for (const search of ['', '?run=r1', '?harness=0']) {
    const { win } = page(search), { fetch, history } = win, { replaceState, pushState } = history;
    assert.equal(install(win), null);
    assert.equal(win.fetch, fetch); assert.equal(history.replaceState, replaceState); assert.equal(history.pushState, pushState);
    assert.equal('__plvHarness' in win, false);
  }
  assert.equal(install(undefined), null); assert.equal(install({}), null);
});

test('with the flag: a frozen window.__plvHarness whose snapshot is plain JSON with every counter, from zero', () => {
  const { win } = page('?harness=1'), hook = install(win), api = win.__plvHarness;
  assert.ok(hook); assert.ok(Object.isFrozen(api)); assert.deepEqual(Object.keys(api), ['version', 'snapshot']); assert.equal(api.version, VERSION);
  const s = api.snapshot();
  assert.deepEqual(JSON.parse(JSON.stringify(s)), s, 'plain JSON');
  assert.equal(s.version, 1); assert.equal(s.timeOrigin, 1700000000000.5); assert.equal(s.now, 5);
  assert.deepEqual(s.run, { id: null, state: null, mode: null, ackState: null, frames: 0, current: -1, iter: null, live: false });
  assert.deepEqual([s.page, s.level, s.stage, s.tiled, s.priors], [null, null, null, null, null]); assert.deepEqual(s.umap, { open: false, width: 0 });
  assert.equal(s.qc, null); assert.equal(s.picker, null, 'no run picker shown');
  assert.deepEqual(s.counters, { stageDraws: 0, umapDraws: 0, timelineDraws: 0, priorRebuilds: 0, priorRebuildMsLast: 0, priorRebuildMsMax: 0,
    priorRebuildMsTotal: 0, priorPolygonsLast: 0, priorWorkerPaints: 0, priorWorkerPaintMsLast: 0, priorWorkerPaintMsMax: 0, priorWorkerPaintMsTotal: 0,
    rangePaints: 0, rangePaintMsLast: 0, rangePaintMsMax: 0, rangePaintMsTotal: 0, rangeKeptLast: null, rangeHiddenLast: null, rangeGreyLast: null,
    analysisApplied: 0, tiledRequests: {}, fetches: 0, fetchInFlight: 0, contextLosses: 0,
    qcApplied: 0, qcAnimFrames: 0, qcAreaEvents: 0, qcSuggestionsShown: 0, qcTrials: 0, qcVerdicts: 0, qcPanelState: null, qcAreasShown: 0,
    qcDeeperOpened: 0, qcDeeperApplied: 0, qcDeeperAnimFrames: 0, qcDeeperRetargets: 0, qcDeeperArrowFrames: 0,
    glowFrames: 0, qcTests: 0, qcTestFetches: 0, qcTestPoolFetches: 0, qcTestViewFetches: 0, qcTestCornerDraws: 0,
    compareOpened: 0, compareDraws: 0, compareFetches: 0, segLayerDraws: 0,
    framesShown: 0 });
  hook.count('stageDraws', 3); hook.tiledRequest('frame'); hook.priorRebuild(25432.17, 41190); hook.priorRebuild(12.34, 5);
  s.counters.tiledRequests.frame = 99;   // a snapshot is a copy
  const c = api.snapshot().counters;
  assert.equal(c.stageDraws, 3); assert.deepEqual(c.tiledRequests, { frame: 1 });
  assert.deepEqual([c.priorRebuilds, c.priorRebuildMsLast, c.priorRebuildMsMax, c.priorRebuildMsTotal, c.priorPolygonsLast], [2, 12.3, 25432.2, 25444.5, 5]);
});

test("a worker's drawing is counted apart from the main thread's rebuild; the mixing check's gauges say what is shown now", () => {
  const hook = createHook(clock());
  hook.priorRebuild(4.2, 6456);                       // the main thread: posting the request and uploading the bitmap
  hook.priorWorkerPaint(56.04, 6456); hook.priorWorkerPaint(31.2); hook.priorWorkerPaint(NaN);
  let c = counters(hook);
  assert.deepEqual([c.priorRebuilds, c.priorRebuildMsMax], [1, 4.2], 'the worker never raises the main-thread maximum');
  assert.deepEqual([c.priorWorkerPaints, c.priorWorkerPaintMsLast, c.priorWorkerPaintMsMax, c.priorWorkerPaintMsTotal], [2, 31.2, 56, 87.2]);
  hook.count('qcApplied'); hook.count('qcApplied'); hook.count('qcAnimFrames', 3); hook.qcShown('live', 2);
  c = counters(hook);
  assert.deepEqual([c.qcApplied, c.qcAnimFrames, c.qcPanelState, c.qcAreasShown], [2, 3, 'live', 2]);
  hook.qcShown('scrubbing', 0); hook.qcShown(null);
  c = counters(hook);
  assert.deepEqual([c.qcPanelState, c.qcAreasShown, c.qcApplied], [null, 0, 2], 'a gauge, not a count; counts stay');
});

test('fetches in flight count until each settles; the caller keeps its own promise; a throwing fetch is not left in flight', async () => {
  const pending = [], { win } = page('?harness=1', { fetch(...args) { if (args[0] === 'throw') throw new TypeError('bad'); const d = deferred(); pending.push(d); return d.promise; } });
  const hook = install(win), native = pending;
  const a = win.fetch('/api/v1/runs'), b = win.fetch('/api/v1/runs/r/xenium');
  assert.equal(a, native[0].promise, 'the very promise the native fetch returned');
  assert.deepEqual([counters(hook).fetches, counters(hook).fetchInFlight], [2, 2]);
  native[0].resolve('ok'); assert.equal(await a, 'ok'); await settle();
  assert.equal(counters(hook).fetchInFlight, 1);
  native[1].reject(Object.assign(new Error('aborted'), { name: 'AbortError' })); await assert.rejects(b, /aborted/); await settle();
  assert.equal(counters(hook).fetchInFlight, 0);
  assert.throws(() => win.fetch('throw'), /bad/);
  assert.deepEqual([counters(hook).fetches, counters(hook).fetchInFlight], [3, 0]);
});

test('URL rewrites keep the flag (a reload keeps the hook); a call without a URL passes through', () => {
  const { win, calls } = page('?harness=1');
  install(win);
  win.history.replaceState(null, '', '?run=' + encodeURIComponent('20261016-120841-sample-1'));
  win.history.replaceState(null, '', '/');
  win.history.pushState({ a: 1 }, '', '/?run=x&harness=1');
  win.history.replaceState({ b: 2 }, '');
  assert.deepEqual(calls, [['replaceState', null, '', '/?run=20261016-120841-sample-1&harness=1'], ['replaceState', null, '', '/?harness=1'],
    ['pushState', { a: 1 }, '', '/?run=x&harness=1'], ['replaceState', { b: 2 }, '']]);
});

test("main.js's view: page, run, level, UMAP pane, stage size, priors, the run picker and frames shown; a failing view is reported, not thrown", async () => {
  const main = await readFile(new URL('main.js', SRC), 'utf8');
  const block = main.slice(main.indexOf('harness?.provide('), main.indexOf('boot().catch('));
  assert.ok(block.length > 100 && main.indexOf('window.sopaCompanion =') < main.indexOf('harness?.provide('), 'after window.sopaCompanion (outside the slices other tests run)');
  const hook = createHook(clock(1));
  const context = vm.createContext({
    harness: hook, counts: { snapshots: 7 }, polygonCount,
    state: { mode: 'run', tool: 'proseg', live: false, current: 1, frames: [{ iter: 0 }, { iter: 250 }],
      run: { id: 'r1', state: 'paused', mode: 'tiled', ack: { state: 'paused', seq: 1 } } },
    // The close popup (shell.mjs ClosePopup.snapshot): shown for a run still sampling, nothing answered yet.
    closePopup: { snapshot: () => ({ open: true, runs: ['r1'], answered: null, sent: false }) },
    // The older runs' prompt (older-runs.mjs snapshot) and a notebook's session (null on the desktop).
    olderPrompt: { snapshot: () => ({ open: false, runs: [], answered: {} }) },
    stage: { canvas: { width: 871, height: 794 } },
    tiled: { plan: { level: 'detail' }, vectorsLoaded: true, requests: { jobs: new Map([['prior', {}], ['frame', {}]]) } },
    // The pane with its Advanced settings: generation 2 asked for and shown, settings changed, Apply not busy.
    analysis: { enabled: true, pane: { style: { width: '681px' } }, gen: 2, applied: { generation: 2, summary: { nclusters: 9 } },
      settings: { changed: true, isBusy: false } },
    priors: { vectors: { cells: [1, 2, 3], nuclei: [1] }, settings: { prior: { on: false }, nuclei: { on: false }, xenium: { on: true } },
      // The sample's segmentation layers (prior-layers.js segmentationsShown).
      segmentationsShown: () => ({ sample: 's1', note: null, loaded: [{ key: 'cellpose_test', name: 'Cellpose', visible: true, color: '#7be29a',
        opacity: 0.8, width: 1, fill: false, state: 'ready', drawn: true }] }) },
    // Compare's view (compare.mjs CompareView.snapshot).
    compareView: { snapshot: () => ({ open: true, state: 'ready', columns: [{ key: 'run', state: 'ready', mixed_per100: 13.2, like100_per100: 11.4,
      points: 4987 }], area: { box: [0, 500, 2100, 2000], size_um: [2100, 1500], tiles: 151 } }) },
    qcClient: { view: { status: 'paused', enabled: true, atStart: false, summary: { seq: 1, iter: 250, shown: 11, per100: 11.3 },
      start: { per100: 9.6 }, areas: [{ L: 'A' }, { L: 'C' }], areaView: 'smoothed', cells: new Uint8Array(3), cellsSeq: 1, cellsCurrent: true } },
    // The overlay draws no area since 2026-10-22: whatever it once counted, the snapshot says none are drawn.
    qcOverlay: { shown: 2 },
    // What the panel shows (its card, trial readout and previous-run comparison) and whether a click is on its way.
    qcPanel: { shown: { card: { rule: 'R4', areas: ['A'], flag: 'cell-compactness', to: 0.02, actions: ['new-run', 'try'], primary: 'new-run' },
      trial: null } }, qcActing: false,
    // Go deeper's window, as qc-deeper.js snapshots it; the test corner, as qc-test-corner.mjs does.
    deeper: { snapshot: () => ({ open: true, cell: 'ffaaaaaa-1', seq: 1, iter: 250, state: 'ready', code: 116, shown: '46' }) },
    testCorner: { snapshot: () => ({ open: true, phase: 'running', frames: 4, shown: { control: { iter: 42, kept: 3, cell_total: 519 }, test: null },
      glow: { on: true, fps: 12 } }) },
    stainPatches: {ownsOutlines:false,snapshot:()=>({active:false}),previewBoxes:()=>context.boxes},
    // Removed areas (stain-exclusions.mjs snapshot): the remove mode, what is on its way and waiting, Undo and Redo.
    stainExclusions: {active:false,snapshot:()=>({active:true,drawing:true,busy:false,loading:false,queued:1,undo:2,redo:0,
      version:3,removed:9,areas:2})},
    // The run picker: what Start still needs (main.js pickerProblems), the boxes on the image and whether they were
    // suggested for the picker as it stands (main.js previewMatches), Start's own state.
    $: id => ({ 'run-kind': { value: 'patches' }, start: { getAttribute: name => name === 'aria-busy' ? context.busy : null } })[id],
    pickerProblems: () => [{ field: 'patch-size', label: 'Size (pixels)', message: 'at most 300, so each patch fits inside the region' },
      { field: null, label: 'Region', message: 'make it at least 20 µm across in both directions' },
      { field: null, label: 'Box 2', message: 'move it inside the region' }, { field: null, label: '', message: 'The sample is still loading.' }],
    boxes: [[0, 0, 64, 64], [64, 0, 128, 64]], matching: false, settled: [],
    previewMatches: settled => { context.settled.push(settled); return context.matching; }, busy: null,
  });
  context.stage.options = { cellColorBy: 'mixing' };
  vm.runInContext(block, context);
  let s = plain(hook.snapshot());
  assert.equal(s.error, undefined, s.error);
  assert.equal(s.glow, null, 'a stage without the glow (glow.mjs) says none');
  context.stage.glow = { on: true, count: 9, fps: 12 };   // Go deeper's suspects glowing (stage.glow)
  s = plain(hook.snapshot());
  assert.deepEqual(s.glow, { on: true, count: 9, fps: 12 });
  // Go deeper's starting outline as the stage keeps it and the veil inside the lit cells (0-1).
  assert.equal(s.startOutline, null); assert.equal(s.innerVeil, 0);
  Object.assign(context.stage, { startOutline: { points: new Float32Array(8), closed: true, count: 4 }, innerLevel: 0.4 });
  assert.deepEqual(plain(hook.snapshot()).startOutline, { count: 4, closed: true }); assert.equal(hook.snapshot().innerVeil, 0.4);
  delete context.stage.startOutline; delete context.stage.innerLevel;
  assert.equal(s.page, 'run'); assert.equal(s.level, 'detail'); assert.deepEqual(s.stage, [871, 794]);
  assert.deepEqual(s.umap, { open: true, width: 681, generation: 2, applied: 2, changed: true, busy: false, nclusters: 9 });
  // Compare's view and the sample's segmentation layers, as their modules give them.
  assert.deepEqual(s.compare, { open: true, state: 'ready', columns: [{ key: 'run', state: 'ready', mixed_per100: 13.2, like100_per100: 11.4,
    points: 4987 }], area: { box: [0, 500, 2100, 2000], size_um: [2100, 1500], tiles: 151 } });
  assert.deepEqual(s.segmentations, { sample: 's1', note: null, loaded: [{ key: 'cellpose_test', name: 'Cellpose', visible: true,
    color: '#7be29a', opacity: 0.8, width: 1, fill: false, state: 'ready', drawn: true }] });
  // The tool the page is in (the header names it) and the close popup, as shown.
  assert.equal(s.tool, 'proseg'); assert.deepEqual(s.closing, { open: true, runs: ['r1'], answered: null, sent: false });
  assert.deepEqual(s.older, { open: false, runs: [], answered: {} }); assert.equal(s.session, null, 'the desktop has no session');
  context.state.session = { state: 'running', sample: { state: 'ready' } }; context.state.ended = null;
  assert.deepEqual(plain(hook.snapshot()).session, { state: 'running', sample: 'ready', ended: null });
  delete context.state.session; delete context.state.ended;
  assert.deepEqual(s.run, { id: 'r1', state: 'paused', mode: 'tiled', ackState: 'paused', frames: 2, current: 1, iter: 250, live: false });
  assert.deepEqual(s.tiled, { vectorsLoaded: true, pending: ['prior', 'frame'] });
  assert.deepEqual(s.priors, { cells: 3, nuclei: 1, drawn: false, layers: { prior: false, nuclei: false, xenium: true } });
  context.priors.shown = { vectors: context.priors.vectors };
  assert.equal(hook.snapshot().priors.drawn, true);
  context.priors.vectors = { ...context.priors.vectors };
  assert.equal(hook.snapshot().priors.drawn, false, 'the preceding tile bitmap is not the current outlines');
  context.priors.shown = { vectors: context.priors.vectors };
  context.state.run.kind = 'stain'; context.state.run.progress = { revision: 2 };
  context.priors.vectors.revision = 1;
  assert.equal(hook.snapshot().priors.drawn, false, 'a newer tile has finished while the preceding vectors are still painted');
  context.priors.vectors.revision = 2;
  assert.equal(hook.snapshot().priors.drawn, true);
  assert.equal(hook.snapshot().exclusions, null, 'removed areas only while their panel is shown');
  context.stainExclusions.active = true;
  assert.deepEqual(plain(hook.snapshot().exclusions), { active: true, drawing: true, busy: false, loading: false, queued: 1,
    undo: 2, redo: 0, version: 3, removed: 9, areas: 2 });
  delete context.state.run.kind;
  assert.equal(hook.snapshot().exclusions, null);
  // The mixing check as shown: the result's frame and iteration (harness metrics compare it with the frame on screen).
  const corner = { open: true, phase: 'running', frames: 4, shown: { control: { iter: 42, kept: 3, cell_total: 519 }, test: null }, glow: { on: true, fps: 12 } };
  const round3 = { card: { rule: 'R4', areas: ['A'], flag: 'cell-compactness', to: 0.02, actions: ['new-run', 'try'], primary: 'new-run' },
    trial: null, acting: false,
    deeper: { open: true, cell: 'ffaaaaaa-1', seq: 1, iter: 250, state: 'ready', code: 116, shown: '46' }, testCorner: corner, testSquare: null };
  // The result's letters stay (the engine's: the harness checks them against the frame on screen); drawn: none.
  assert.deepEqual(s.qc, { state: 'paused', enabled: true, seq: 1, iter: 250, snapped: null, later: false, shown: 11, atStart: false, start: 9.6,
    areas: ['A', 'C'], areaView: 'smoothed', drawn: 0, colours: true, cells: true, cellsSeq: 1, cellsCurrent: true, ...round3 });
  // Looking back: the iteration the badge names (the result of another frame), and whether the result is a later frame's
  // (the oldest after a frame before every result: the harness's "ahead" count leaves those out).
  context.qcClient.view = { ...context.qcClient.view, snapped: 250, later: true };
  assert.deepEqual([plain(hook.snapshot()).qc.snapped, plain(hook.snapshot()).qc.later], [250, true]);
  context.qcClient.view = { ...context.qcClient.view, snapped: null, later: false };
  // The test's square on the overlay while a test exists (QcOverlay.testSquare), null otherwise.
  context.qcOverlay.testSquare = [100, 200, 520, 620];
  assert.deepEqual(plain(hook.snapshot()).qc.testSquare, [100, 200, 520, 620]);
  delete context.qcOverlay.testSquare;
  context.qcClient.view = { ...context.qcClient.view, atStart: true, summary: null, areas: [], areaView: 'start', cells: null, cellsSeq: null,
    cellsCurrent: false };
  context.qcPanel = {};   // a panel that shows no card, readout or comparison
  assert.deepEqual(plain(hook.snapshot()).qc, { state: 'paused', enabled: true, seq: null, iter: null, snapped: null, later: false, shown: 10, atStart: true, start: 9.6,
    areas: [], areaView: 'start', drawn: 0, colours: true, cells: false, cellsSeq: null, cellsCurrent: false, card: null, trial: null,
    acting: false, deeper: { open: true, cell: 'ffaaaaaa-1', seq: 1, iter: 250, state: 'ready', code: 116, shown: '46' },
    testCorner: corner, testSquare: null });
  // The test of a setting as shown (the harness's qc.test, qc-panel.js testShown), while the section shows something:
  // the refusals, waits and caveats by their codes.
  const shown = { mode: 'running', id: 't1', cell: 'ffaaaaaa-1', flag: 'diffusion-probability', value: 0.3, current: 0.2, iter: 212, of: 450, runs: 8,
    done: 0, phase: 'running', corner: true, start: 'cells', area: 'crop', reasons: [], waits: [], caveats: ['lasso', 'too_few'],
    pool: { cells: 23, suspects: 187, need: 160 }, verdict: null, rule: null, hold: null };
  context.qcPanel = { testShown: shown };
  assert.deepEqual(plain(hook.snapshot()).qc.test, shown);
  assert.deepEqual(Object.keys(hook.snapshot().qc.test), ['mode', 'id', 'cell', 'flag', 'value', 'current', 'iter', 'of', 'runs', 'done', 'phase',
    'corner', 'start', 'area', 'reasons', 'waits', 'caveats', 'pool', 'verdict', 'rule', 'hold']);
  context.qcPanel = { testShown: { ...shown, mode: 'waiting', phase: 'waiting', area: null, waits: ['not_at_end'], caveats: [], pool: null } };
  assert.deepEqual(plain(hook.snapshot()).qc.test.waits, ['not_at_end']);
  context.qcPanel = {}; assert.equal('test' in hook.snapshot().qc, false, 'nothing shown: no test');
  assert.equal(s.counters.framesShown, 7);
  assert.equal(s.picker, null, 'a run is open: no picker');
  Object.assign(context.state, { mode: 'pick', run: null, frames: [], current: -1 }); context.analysis.enabled = false; context.priors.vectors = null;
  context.state.tool = 'stain';
  const picker = hook.snapshot();
  assert.deepEqual([picker.page, picker.level, picker.tiled, picker.run.id, picker.run.frames, picker.umap.open, picker.priors.cells], ['pick', null, null, null, 0, false, 0]);
  assert.equal(picker.tool, 'stain');
  // The run picker: problems by field id (where there is no field, its label and words as a press lists them, so a box
  // keeps its number), the boxes on the image, how many of them were suggested for the picker as it stands with nothing
  // waiting (0 otherwise), Start busy or not.
  const boxes = [[0, 0, 64, 64], [64, 0, 128, 64]];
  assert.deepEqual(plain(picker.picker), { kind: 'patches', problems: ['patch-size', 'Region: make it at least 20 µm across in both directions',
    'Box 2: move it inside the region', 'The sample is still loading.'], preview: 0, boxes, busy: false },
    'boxes of earlier settings, or a change waiting for its own: shown, not counted');
  assert.deepEqual(context.settled, [true], 'asked with nothing still waiting');
  context.matching = true; context.busy = 'true'; context.pickerProblems = () => [];
  assert.deepEqual(plain(hook.snapshot().picker), { kind: 'patches', problems: [], preview: 2, boxes, busy: true });
  context.boxes = null;
  assert.deepEqual(plain(hook.snapshot().picker), { kind: 'patches', problems: [], preview: 0, boxes: null, busy: true }, 'no boxes on the image');
  context.state.mode = 'run'; context.state.run = { id: 'd', mode: 'detail', state: 'running' };
  assert.equal(hook.snapshot().run.mode, 'detail');
  Object.assign(context.state, { mode: 'empty', run: null, tool: null });
  assert.equal(hook.snapshot().tool, null, "the first screen's two cards: no tool");
  context.stage = null;
  const broken = hook.snapshot();
  assert.match(broken.error, /canvas/); assert.equal(broken.counters.stageDraws, 0);
});

test('PriorLayers: one rebuild per changed key, timed over building and stroking the paths only, with the polygons stroked', async () => {
  const perf = clock(), hook = createHook(perf), { PriorLayers } = await withHook('prior-layers.js', hook);
  const g = new Proxy({ setTransform() { perf.t += 100; }, drawImage() { perf.t += 50; }, stroke() { perf.t += 2; } },
    { get: (o, k) => o[k] || (() => {}), set: () => true });
  // The paths are Path2D objects now (prior-paths.mjs): a polygon's own path is built once, into its grid cell's tile.
  globalThis.Path2D = class { moveTo() { perf.t += .5; } lineTo() { perf.t += .5; } closePath() {} addPath() {} };
  globalThis.document = { createElement: () => ({ width: 0, height: 0, getContext: () => g }) };
  // PriorLayers times its drawings by the page clock too (whether one more fits while the camera moves): this one, on
  // which every drawing costs over 100 ms (the transform), so a moving camera is never drawn again here.
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'performance');
  Object.defineProperty(globalThis, 'performance', { value: { now: () => perf.t, measure() {}, clearMeasures() {} }, configurable: true, writable: true });
  const stage = { canvas: { width: 800, height: 600 }, camera: { x: 0, y: 0, zoom: 1 }, ratio: 1, info: { cells: 3 }, orbit: { active: false }, gl: fakeGL(), imageProgram: {} };
  const priors = new PriorLayers(stage, () => 'rgb(1,2,3)');
  try {
    const square = { poly: [[0, 0], [1, 0], [1, 1], [0, 1]] };
    priors.reset(); priors.settings.xenium.on = true;
    priors.vectors = { cells: [square, square, square], nuclei: [square, square] };
    priors.draw();
    assert.deepEqual([counters(hook).priorRebuilds, counters(hook).priorPolygonsLast], [1, 3]);
    assert.equal(counters(hook).priorRebuildMsLast, 3 * 4 * .5 + 2, '12 path calls and one stroke; not the transform');
    priors.draw();
    assert.equal(counters(hook).priorRebuilds, 1, 'an unchanged key reuses the canvas');
    // While the camera moves or the stage is resized (Stage.emit sets movingUntil), the last raster is reused.
    stage.movingUntil = performance.now() + 60000;
    stage.camera.x = 5; priors.draw();
    stage.canvas.width = 500; priors.draw();   // a narrower stage (the UMAP pane opened)
    assert.equal(counters(hook).priorRebuilds, 1, 'no rebuild while the camera or the stage size is changing');
    stage.movingUntil = 0; priors.draw();
    assert.deepEqual([counters(hook).priorRebuilds, counters(hook).priorRebuildMsLast], [2, 2], 'once settled: one stroke, the tile reused');
    priors.settings.nuclei.on = true; priors.draw();
    const c = counters(hook);
    assert.deepEqual([c.priorRebuilds, c.priorPolygonsLast, c.priorRebuildMsLast, c.priorRebuildMsMax, c.priorRebuildMsTotal], [3, 5, 2 + 2 * 4 * .5 + 2, 8, 8 + 2 + 8]);
    priors.vectorsVisible = false; stage.camera.x = 9; priors.draw();
    assert.equal(counters(hook).priorRebuilds, 3, 'nothing visible: no rebuild');
  } finally { priors.reset(); delete globalThis.document; delete globalThis.Path2D; Object.defineProperty(globalThis, 'performance', saved); }
});

test('Stage counts each draw and each lost context', async () => {
  const hook = createHook(clock()), { Stage } = await withHook('stage.js', hook), loop = fakeLoop(), listeners = {};
  const canvas = { width: 0, height: 0, addEventListener(type, fn) { listeners[type] = fn; }, getContext: () => fakeGL(), getBoundingClientRect: () => ({ width: 1000, height: 800 }) };
  const stage = loop.with(() => new Stage(canvas));
  loop.run(100);
  loop.with(() => stage.panBy(5, 3)); loop.run(200);
  assert.ok(stage.drawCount >= 2); assert.equal(counters(hook).stageDraws, stage.drawCount);
  loop.with(() => listeners.webglcontextlost({ preventDefault() {} }));
  assert.equal(counters(hook).contextLosses, 1);
});

test('Timeline counts each draw', async () => {
  const hook = createHook(clock()), { Timeline } = await withHook('timeline.js', hook), loop = fakeLoop();
  const timeline = loop.with(() => new Timeline(fakeElement({ clientWidth: 1000, clientHeight: 116 }), { onScrub() {} }));
  loop.run(100);
  loop.with(() => { timeline.setPlan([{ phase: 'burnin', frames: 40 }]); timeline.setCurrent(3, true); });
  loop.run(400);
  assert.ok(timeline.drawCount >= 2); assert.equal(counters(hook).timelineDraws, timeline.drawCount);
});

test('UmapView counts each render and each lost context', async () => {
  const hook = createHook(clock()), { UmapView } = await withHook('umap.js', hook), frames = new Map();
  class Node {
    constructor(tag) { this.tagName = tag; this.children = []; this.listeners = {}; this.style = {}; this.parentElement = null; }
    append(...nodes) { for (const n of nodes) { n.parentElement = this; this.children.push(n); } }
    remove() {} addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); } removeEventListener() {}
    emit(type) { for (const fn of this.listeners[type] || []) fn({ preventDefault() {} }); }
    getBoundingClientRect() { return { left: 0, top: 0, width: 300, height: 150 }; }
  }
  const canvas = Object.assign(new Node('canvas'), { width: 300, height: 150, offsetLeft: 0, offsetTop: 0, getContext: () => fakeGL(), setPointerCapture() {} });
  new Node('section').append(canvas);
  const globals = { document: { createElement: tag => new Node(tag), body: new Node('body') }, window: Object.assign(new Node('window'), { devicePixelRatio: 1 }),
    requestAnimationFrame: fn => { frames.set(frames.size + 1, fn); return frames.size; }, cancelAnimationFrame: id => frames.delete(id),
    performance: { now: () => 0 }, ResizeObserver: class { observe() {} disconnect() {} } };
  const saved = Object.fromEntries(Object.keys(globals).map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  try {
    const view = new UmapView(canvas);
    view.setData({ clusters: Int16Array.of(0, 1, 1), coords: Float32Array.of(0, 0, 0, 1, 1, 0, 2, 0, 0) });
    for (let n = 0; frames.size && n < 100; n++) { const due = [...frames.values()]; frames.clear(); due.forEach(fn => fn(16 * n + 1000)); }
    assert.ok(view.drawCount >= 1); assert.equal(counters(hook).umapDraws, view.drawCount);
    canvas.emit('webglcontextlost');
    assert.equal(view.lost, true); assert.equal(counters(hook).contextLosses, 1);
  } finally { for (const [k, d] of Object.entries(saved)) { if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k]; } }
});

test('Analysis counts a result applied once, and not clearing it', async () => {
  const hook = createHook(clock()), { Analysis } = await withHook('analysis.js', hook);
  const a = Object.assign(Object.create(Analysis.prototype), { applied: null, appliedAt: -Infinity, coloured: true, hoverLabel: 0, hoverCluster: null, hidden: new Set(),
    stage: { setAnalysis() {}, set() {}, clusterGPU: null }, panel: { render() {} }, controls: {}, empty: {}, legend: { replaceChildren() {} },
    view: { setData() {}, highlightCells() {}, highlightCluster() {} }, legendFor: () => ({ update() {} }) });
  const result = { seq: 3, clusters: Int16Array.of(0, 1), coords: new Float32Array(6), summary: { iter: 3, clusters: [], ms: {} } };
  a.apply(result); a.apply(result);
  assert.equal(counters(hook).analysisApplied, 1, 'the same result again is no new application');
  a.apply(null); a.apply({ ...result, seq: 4 });
  assert.equal(counters(hook).analysisApplied, 2);
});

test('TiledLoader counts requests by kind when they start: identical requests share one, one cancelled before it starts is none', async () => {
  const hook = createHook(clock()), { ViewRequests } = await withHook('tiled-loader.mjs', hook), requests = new ViewRequests(), d = deferred();
  const first = requests.run('frame', 'a', () => d.promise);
  assert.equal(requests.run('frame', 'a', () => { throw new Error('not started twice'); }), first);
  requests.run('image', 'x', async () => 'never').catch(() => {}); requests.cancel('image');
  requests.run('vectors', 'vectors', async () => ({ cells: [] }));
  await settle();
  assert.deepEqual(counters(hook).tiledRequests, { frame: 1, vectors: 1 });
  requests.run('frame', 'b', async () => 'next'); first.catch(() => {});   // a new box replaces the frame in flight
  await settle();
  assert.deepEqual(counters(hook).tiledRequests, { frame: 2, vectors: 1 });
});

test('live range paint times are counted independently from ordinary outline painting',()=>{
  const hook=createHook(clock());hook.rangePaint(20.34);hook.rangePaint(10);hook.rangePaint(NaN);
  const c=counters(hook);assert.equal(c.rangePaints,2);assert.equal(c.rangePaintMsMax,20.3);assert.equal(c.rangePaintMsTotal,30.3);
  assert.equal(c.priorWorkerPaints,0);
  // What each paint left (prior-layers.js installed): the harness reads the last kept, hidden and grey counts.
  hook.rangePaint(5,12,{kept:9,hidden:3,grey:0});const d=counters(hook);
  assert.deepEqual([d.rangeKeptLast,d.rangeHiddenLast,d.rangeGreyLast,d.rangePaints],[9,3,0,3]);
  hook.rangePaint(5);const e=counters(hook);assert.deepEqual([e.rangeKeptLast,e.rangeHiddenLast,e.rangeGreyLast],[null,null,0]);
});
