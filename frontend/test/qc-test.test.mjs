import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { domFixture, settle } from './dom-fixture.mjs';
import { QC_TEXT as C, bannedWords, settingChecked, copy, settingValue, TEST_REASON_WORDS, TEST_REASON_KINDS, testReasonText, TEST_REASON_GROUPS,
  testReasonGroup, TEST_RUN_REFUSALS } from '../src/qc-copy.mjs';
import { EXPLAIN_TEXT } from '../src/explain.mjs';
import { QcPanel, TEST_FLAGS, TEST_TRIED, TEST_RANGES, TEST_STATES, TEST_ACTIVE, TEST_PHASES, TEST_ACTIONS, TEST_STARTS, TEST_RUNS_PER_SIDE,
  TEST_POLL_MS, TEST_POOL_SHOW_MS, TEST_POOL_WAIT_S, TEST_POOL_AGAIN_MS, TEST_POOL_RUN_WAIT_MS, TEST_FIELDS, TEST_WENT, TEST_COST, TEST_AREAS,
  testRecord, testStart, testName, holdOf, driftOff, testReasons, testWaits, testCaveats, reasonCodes, testNotHere, poolLine, areaText, runsText,
  testProblems, tenths, percent, percentPair, wholePercents, calibrationText, calibrationShortText, earlierWord, testReadout, testProgress, testWaiting,
  holdGiven, holdAtEnd, TEST_KINDS, TEST_LAYER_COST, testValueText, testEstimate, layerFactor } from '../src/qc-panel.js';

// "Test a setting here…": the pool of nearby cells with the same pattern the engine fixes before any run (its preview,
// GET /runs/{id}/qc/test/pool, with its refusals, what a test would wait for first and where it was not measured),
// fresh runs as now and with the value, and the engine's answer in five parts (result v2). The panel against a minimal
// DOM and a scripted page: the form (one order of settings for every start, nothing prefilled, far drift not tried
// yet), the pool asked for once per cell and end of the run, Run waiting for it and posting {cell, flag, value} only
// (also while waits hold: the test then waits, and wherever it was not measured), the refusals listed and the waits and
// caveats in tooltips (the engine's codes in the page's words: the shared fixture), the answer in the deck's words for
// every example of the engine's answers (the shared fixture), New run with this, Keep, Run again with the earlier
// answer's word; then main.js's side.

const REASONS = JSON.parse(readFileSync(new URL('./fixtures/qc-test-reasons.json', import.meta.url), 'utf8'));
const EXAMPLES = JSON.parse(readFileSync(new URL('./fixtures/qc-test-result.json', import.meta.url), 'utf8'));
const RESULTS = Object.fromEntries(EXAMPLES.map(e => [e.name, e.result]));

const f32 = x => new Float32Array([x])[0];          // proseg reports knobs as 32-bit floats
const SETTINGS = { 'cell-compactness': { label: 'Cell compactness (lower = rounder cells)', value: f32(0.04), min: 0.001, max: 10 },
  'prior-weight': { label: 'Prior weight (higher = the starting cells hold their transcripts harder)', value: 1, min: 0, max: 1000 },
  'diffusion-probability': { label: 'Share of far-drifted transcripts (diffusion probability)', value: f32(0.2), min: 0, max: 1 } };
const NAMES = ['Cell compactness', 'Prior weight', 'Share of far-drifted transcripts'];   // the Live knobs names a Setting line may hold
const END = { schedule_done: { iter: 450, seq: 451 } };   // a run waiting at the end of its schedule
const PAIRED = { prior: { cells: { kind: 'shapes', key: 'cell_boundaries' }, nuclei: { kind: 'shapes', key: 'nucleus_boundaries' } }, sample_kind: 'zarr',
  prepared: { seeded_from: 'nuclei' }, ...END };
const CELLS = { prior: { cells: { kind: 'shapes', key: 'cellpose_boundaries' }, nuclei: null }, sample_kind: 'zarr', prepared: { seeded_from: 'cells' }, ...END };
const SQUARE = { rule: 'square', box: [3150, 3150, 3650, 3650], inner: [3175, 3175, 3625, 3625] };
const REGION = { rule: 'as_is', box: [100, 100, 400, 400], inner: [125, 125, 375, 375] };
/** A pool reply (GET /runs/{id}/qc/test/pool): the pool reached; `reasons` a refusal's (codes, or {code, ...data}), its
 *  pool then null; `waits` and `caveats` none (OFFERED gives them). */
const POOL = (extra = {}) => ({ v: 1, state: 'ready', run: 'r1', cell: 6, cell_name: 'ffaaaaaa-7', seq: 451, iter: 450, model: 'm1', constants: 'c1',
  start: 'paired', runs_per_side: 3, k: 160, area: SQUARE, offer: true, reasons: [], waits: [], caveats: [], row: { start: 'paired', situation: 'square' },
  pattern: 'own',
  line: { likely: 12, own: 9, none: 3, neighbour: 0, other: 0 }, this_cell: { likely: 12, suspects: 9 },
  pool: { cells: 23, suspects: 187, need: 160, reached: true, available: null, members: [] }, estimate_s: 45, ms: { total: 1180.4 }, message: null, ...extra });
const reason = r => typeof r === 'string' ? { code: r, words: "the engine's own words" } : r;
const REFUSED = (reasons, extra = {}) => POOL({ offer: false, reasons: reasons.map(reason), pattern: null, line: null, this_cell: null, pool: null, ...extra });
/** A pool reply that offers the test with `waits` (what a press waits for first) and `caveats` (where it was not
 *  measured): codes, or {code, ...data}. */
const OFFERED = (waits = [], caveats = [], extra = {}) => POOL({ waits: waits.map(reason), caveats: caveats.map(reason), ...extra });
/** The fixture's codes of one kind, and each as the engine gives it ({code, words, ...data}). */
const OF_KIND = kind => REASONS.filter(e => e.kind === kind);
const GIVEN = e => ({ code: e.code, words: "the engine's own words", ...e.data });
/** A test as the engine gives it (its {test} record). */
const PROGRESS = (extra = {}) => ({ iter: 0, of: 450, runs: 6, done: 0, sides: { A: 0, B: 0 }, marks: [200, 400], phase: 'preparing', retry: false, ...extra });
const RECORD = (extra = {}) => ({ id: 't1', state: 'preparing', cell: 6, cell_name: 'ffaaaaaa-7', flag: 'cell-compactness', name: 'compactness', value: 0.02,
  current: 0.04, tried: true, start: 'paired', area: SQUARE, square: SQUARE.box, runs_per_side: 3, threads: 8,
  runs: { A1: null, A2: null, A3: null, B1: null, B2: null, B3: null }, progress: PROGRESS(),
  pool: { cells: 23, suspects: 187, k: 160, matched: null, uniform: null, unsure: null, not_found: null }, message: null, result: null,
  was_running: false, seconds: {}, hold: null, no_factorization: false, reruns: 0, estimate_s: 45, waiting: [], reasons: [], caveats: [],
  row: { start: 'paired', situation: 'square' }, ...extra });
/** A test waiting for `waits` (codes, or {code, ...data}) as the engine gives it: no area, square or pool yet, its runs
 *  without roles. */
const WAITING = (waits = ['not_at_end'], extra = {}) => RECORD({ state: 'waiting', area: null, square: null, pool: null,
  runs: { A1: null, A2: null, A3: null, B1: null, B2: null, B3: null }, progress: PROGRESS({ phase: 'waiting' }), waiting: waits.map(reason), ...extra });
/** A test that is done with the example `name` of the engine's answers as its result. */
const DONE = (name = 'not_clear_apart', extra = {}) => {
  const r = RESULTS[name];
  return RECORD({ state: 'done', flag: r.setting.flag, value: r.setting.value, current: r.setting.current, start: r.start, area: r.situation === 'square' ? SQUARE : REGION,
    square: r.situation === 'square' ? SQUARE.box : null, runs_per_side: r.constants.runs_per_side,
    progress: PROGRESS({ iter: 450, done: 2 * r.constants.runs_per_side, sides: { A: 450, B: 450 }, phase: 'reading' }), result: r, ...extra });
};
const notFound = () => Object.assign(new Error('No test for this run'), { status: 404 });
const refusal = (status, data = null) => Object.assign(new Error("the engine's own words"), { status, data });
/** A share as the page says it, worked out here on its own (the fixture's shares come to 4 decimals). */
const pc = x => Math.round(Math.round(1000 * x) / 10);

/** The panel in a details host on a page with a scripted engine: `env.previews` / `env.started` / `env.fetched` /
 *  `env.cancelled` / `env.answered` / `env.newRuns` record the page's calls, `env.reply` scripts them (functions
 *  returning a value or a promise); `env.models` every model the corner was handed, `env.model()` the last. */
function page(t, { cell = 7, run = 'paused', status = 'live', enabled = true, wire = true, preview = true, record = PAIRED, settings = SETTINGS } = {}) {
  const { document } = domFixture();
  document.hidden = false;
  const env = { document, now: 0, next: 1, tasks: new Map(), frames: 0, previews: [], started: [], fetched: 0, cancelled: [], answered: [], newRuns: [],
    shows: 0, models: [],
    reply: { pool: () => POOL(), start: () => RECORD(), fetch: () => RECORD(), cancel: () => null, answer: () => null, newRun: () => null } };
  const add = (fn, ms, kind) => { const id = env.next++; env.tasks.set(id, { fn, at: env.now + ms, kind }); return id; };
  const scheduler = { now: () => env.now, setTimeout: (fn, ms) => add(fn, ms, 'timer'), clearTimeout: id => env.tasks.delete(id),
    requestAnimationFrame: fn => add(fn, 1, 'frame'), cancelAnimationFrame: id => env.tasks.delete(id) };
  /** Time passes: the timers and frames due run in order, each followed by the promises it started. */
  env.advance = async ms => {
    const until = env.now + ms;
    for (let n = 0; ; n++) {
      assert.ok(n < 200, 'nothing loops by itself');
      const next = [...env.tasks].sort((a, b) => a[1].at - b[1].at)[0];
      if (!next || next[1].at > until) break;
      env.tasks.delete(next[0]); env.now = next[1].at; next[1].fn(env.now); await settle(); await settle();
    }
    env.now = until; await settle(); await settle();
  };
  const globals = { document, window: {}, matchMedia: () => ({ matches: false }) };
  const original = new Map(Object.keys(globals).map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { value: v, writable: true, configurable: true });
  t.after(() => { for (const [k, d] of original) if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k]; });
  env.host = document.createElement('details'); env.host.open = true; document.body.append(env.host);
  const calls = !wire ? {} : {
    testSetting: flag => settings[flag] ?? null,
    onTestPreview: (label, options) => { env.previews.push([label, options?.wait, typeof options?.signal?.aborted === 'boolean']); return env.reply.pool(label, options); },
    onTestStart: (...args) => { env.started.push(args); return env.reply.start(...args); },
    onTestFetch: () => { env.fetched++; return env.reply.fetch(); },
    onTestCancel: (...args) => { env.cancelled.push(args); return env.reply.cancel(args[0]); },
    onTestAnswer: (...args) => { env.answered.push([args[0].id, ...args.slice(1)]); return env.reply.answer(args[0]); },
    onTestNewRun: test => { env.newRuns.push(test.id); return env.reply.newRun(test); },
    onTestShow: () => { env.shows++; },
    onTestCorner: model => env.models.push(model) };
  if (!preview) delete calls.onTestPreview;
  env.panel = new QcPanel(env.host, { scheduler, storage: null, onFrame: () => env.frames++, ...calls });
  t.after(() => env.panel.dispose());
  env.state = (extra = {}) => env.panel.setState({ state: status, enabled, runId: 'r1', model: 'm1', series: [], ...extra });
  env.state();
  env.run = (extra = {}) => env.panel.setTestRun({ id: 'r1', state: run, ...record, ...extra });
  env.run();
  if (cell) env.panel.setCell({ label: cell, name: `ffaaaaaa-${cell}`, color: '#123', line: '' });
  env.box = env.panel.testBox;
  env.model = () => env.models.at(-1) ?? { open: false };
  env.form = () => env.model().form;
  env.act = (action, payload) => env.panel.testAction(action, payload);
  env.find = action => env.box.all().find(n => n.dataset?.action === action);
  env.status = () => env.box.all().find(n => n.className?.includes('qc-test-status'));
  env.open = async () => { env.find('test-open').click(); await settle(); await settle(); };
  /** The words the section shows, one line per node that holds words, hidden ones left out. */
  env.lines = (node = env.box) => node.hidden ? [] : [...(node.text ? [node.text] : []), ...node.childNodes.flatMap(n => env.lines(n))];
  return env;
}
/** Every word a model holds (its data left out: the cell's name, the run, flags, phases, codes and the engine's verdict),
 *  with a Setting line's exact Live knobs name taken out, as the harness reads the page. */
const modelWords = model => {
  const out = [];
  const walk = (value, key) => {
    if (['cell', 'run', 'flag', 'phase', 'start', 'key', 'busy', 'field', 'code', 'word', 'why', 'verdict', 'rule', 'side', 'state', 'fields'].includes(key)) return;
    if (typeof value === 'string') out.push(settingChecked(value, NAMES));
    else if (Array.isArray(value)) value.forEach(v => walk(v, null));
    else if (value && typeof value === 'object') for (const [k, v] of Object.entries(value)) walk(v, k);
  };
  walk(model, null);
  return out;
};
const banned = env => bannedWords([...env.lines(), ...env.box.all().map(n => n.title ?? ''), ...env.models.flatMap(modelWords)].join('\n'));
/** The test's own words hold no clash word (each means something else in this panel) and "hold" only in the hold's name. */
const CLASH = /look-alike|\bsimilar\b|same kind|like for like/i;
const clashFree = text => !CLASH.test(text) && !/\bhold\b/i.test(String(text).replace(C.testHold, ''));

test('the helpers: the settings, the test as the engine gives it, the form\'s lines and what a press of Run still needs', () => {
  assert.deepEqual(TEST_FLAGS, ['cell-compactness', 'prior-weight', 'diffusion-probability', 'burnin-voxel-size', 'ab-nihlo-bubble-prob',
    'no-diffusion', 'diffusion-sigma-near', 'density-bins', 'voxel-layers'], 'one order, whatever the start');
  assert.deepEqual(TEST_TRIED, { 'cell-compactness': true, 'prior-weight': true, 'diffusion-probability': false, 'burnin-voxel-size': false,
    'ab-nihlo-bubble-prob': false, 'no-diffusion': false, 'diffusion-sigma-near': false, 'density-bins': false, 'voxel-layers': false });
  assert.deepEqual(TEST_RANGES, { 'diffusion-probability': [0.1, 0.3] });
  assert.deepEqual(Object.entries(TEST_KINDS).filter(([, k]) => k !== 'number'),
    [['burnin-voxel-size', 'choice'], ['no-diffusion', 'switch'], ['density-bins', 'whole'], ['voxel-layers', 'whole']]);
  // The engine's twins, through the shared fixture.
  const twins = JSON.parse(readFileSync(new URL('./fixtures/qc-test-settings.json', import.meta.url), 'utf8'));
  assert.deepEqual([twins.flags, twins.tried, twins.kinds, twins.ranges, twins.layer_cost],
    [[...TEST_FLAGS], { ...TEST_TRIED }, { ...TEST_KINDS }, { 'diffusion-probability': [0.1, 0.3] }, TEST_LAYER_COST]);
  // A tested value in words: with its unit, the switch as drift off or on.
  assert.deepEqual([testValueText('burnin-voxel-size', 1), testValueText('no-diffusion', 1), testValueText('no-diffusion', 0, 'short'),
    testValueText('density-bins', 8), testValueText('cell-compactness', f32(0.02))], ['1 µm', 'drift off', 'on', '8', '0.02']);
  assert.equal(runsText(4, 1, 50, 'burnin-voxel-size'), 'Your run waits · 4 runs as now, 4 with 1 µm · about 50 s');
  assert.equal(runsText(4, 1, null, 'no-diffusion'), 'Your run waits · 4 runs as now, 4 with drift off');
  assert.deepEqual([testEstimate(50, 'voxel-layers', 8, 4), testEstimate(50, 'density-bins', 8, 5), layerFactor(2)], [Math.round(50 * layerFactor(8)), 50, 1]);
  // A test waits first (for the run's last iteration, the mixing check, another run), then its runs work: both on its way.
  assert.deepEqual(TEST_STATES, ['waiting', 'preparing', 'running', 'comparing', 'done', 'failed', 'cancelled']);
  assert.deepEqual(TEST_ACTIVE, ['waiting', 'preparing', 'running', 'comparing']); assert.deepEqual(TEST_PHASES, ['form', 'starting', ...TEST_STATES]);
  assert.deepEqual(TEST_ACTIONS, ['setting', 'value', 'run', 'cancel', 'new', 'keep', 'again', 'close']);
  // A start from nuclei only is judged with the nuclei-and-cells rows: 3 a side, as they.
  assert.deepEqual(TEST_STARTS, ['cells', 'paired', 'nuclei']); assert.deepEqual(TEST_RUNS_PER_SIDE, { paired: 3, cells: 4, nuclei: 3 });
  assert.deepEqual([TEST_POLL_MS, TEST_POOL_SHOW_MS, TEST_POOL_WAIT_S, TEST_POOL_AGAIN_MS, TEST_POOL_RUN_WAIT_MS], [1000, 250, 5, 2000, 30000]);
  assert.deepEqual(TEST_FIELDS, { flag: 'qc-test-flag', value: 'qc-test-value' });
  assert.deepEqual(TEST_WENT, ['stay', 'neighbour', 'other', 'background']); assert.deepEqual(TEST_COST.map(c => c[0]), ['own', 'all', 'perCell']);
  // The settings by their names: compactness only by Live knobs' own name, the hold and far drift by their plain ones.
  assert.deepEqual(TEST_FLAGS.map(testName), ['Cell compactness', 'Hold to the starting cells', 'Far drift', 'Starting grid (µm)', 'Background gaps',
    'Drift (on or off)', 'Short drift distance (µm)', 'Crowding levels', 'Depth layers']); assert.equal(testName('dispersion'), null);
  // The test itself, or under `test`; anything else is none.
  const record = RECORD();
  assert.equal(testRecord(record), record); assert.equal(testRecord({ test: record }), record);
  const waiting = WAITING(); assert.equal(testRecord({ test: waiting }), waiting, 'a test waiting for its run\'s end is one');
  for (const bad of [null, undefined, 'done', {}, { state: 'queued' }, { test: { state: 'over' } }]) assert.equal(testRecord(bad), null);
  // How the run started: the test's own word first, else the run's record by the engine's rule.
  assert.equal(testStart(PAIRED), 'paired'); assert.equal(testStart(CELLS), 'cells'); assert.equal(testStart(null), null);
  assert.equal(testStart({ prior: { cells: null, nuclei: { kind: 'shapes', key: 'n' } }, sample_kind: 'zarr' }), 'nuclei');
  assert.equal(testStart(PAIRED, { start: 'cells' }), 'cells'); assert.equal(testStart(CELLS, { start: 'other' }), 'cells');
  // How firmly the run holds its starting cells (the harness's snapshot): its record's, else the test's word.
  assert.deepEqual(holdOf({ options: { 'prior-weight': 0.5 }, info: { prior_weight: 6.9 } }), { value: 0.5, effective: 6.9 });
  assert.deepEqual(holdOf({ info: { prior_weight: 6.9 } }), { value: 1, effective: 6.9 }); assert.deepEqual(holdOf(null, { hold: { value: 1, effective: 1.82 } }), { value: 1, effective: 1.82 });
  assert.equal(holdOf(null, null), null);
  // Drift switched off in the run (its options, its live values): far drift does nothing.
  assert.equal(driftOff({ options: { 'no-diffusion': true } }), true); assert.equal(driftOff({ options: { 'no-diffusion': true }, live_values: { 'no-diffusion': false } }), false);
  assert.equal(driftOff({ live_values: { 'no-diffusion': true } }), true); assert.equal(driftOff(null), false);
  // The pool line, its numbers kept as data and what a test needs in its tooltip (a pool short of it is tested all the
  // same: 'short' stays data); the area (the region as it is, a square, or a window cut to a narrower region); the runs.
  const tip = 'The test was measured with 160 suspects or more. With fewer it runs all the same, and its answer says so.';
  assert.deepEqual(poolLine({ cells: 23, suspects: 187, need: 160, reached: true }),
    { state: 'reached', text: 'Nearby cells with the same pattern: 23 cells, 187 suspects', tooltip: tip, cells: 23, suspects: 187, need: 160 });
  assert.deepEqual(poolLine({ cells: 1, suspects: 9, need: 160, reached: false }), { state: 'short', text: 'Nearby cells with the same pattern: 1 cell, 9 suspects',
    tooltip: tip, cells: 1, suspects: 9, need: 160 });
  assert.equal(poolLine({ cells: 1234, suspects: 1500, need: 1600 }).text, 'Nearby cells with the same pattern: 1,234 cells, 1,500 suspects');
  assert.equal(poolLine({ cells: 1234, suspects: 1500, need: 1600 }).tooltip, tip.replace('160', '1,600'));
  assert.equal(poolLine({ cells: 2, suspects: 170, need: 160, reached: false }).state, 'short', 'the engine says it fell short');
  for (const bad of [null, {}, { cells: 2, suspects: '9', need: 160 }]) assert.equal(poolLine(bad), null);
  assert.deepEqual(TEST_AREAS, ['as_is', 'square', 'crop']);
  assert.equal(areaText(SQUARE), 'Area: a 500 × 500 µm square around this cell'); assert.equal(areaText(REGION), 'Area: your whole region, 300 × 300 µm');
  assert.equal(areaText({ rule: 'as_is', box: [0, 0, 1200.4, 449.6] }), 'Area: your whole region, 1,200 × 450 µm');
  assert.equal(areaText({ rule: 'crop', box: [3150, 3200.2, 3650, 3379.8] }), 'Area: 500 × 180 µm around this cell');
  for (const bad of [null, { rule: 'lasso', box: [0, 0, 1, 1] }, { rule: 'square', box: [0, 0, 1] }, { rule: 'crop', box: null }]) assert.equal(areaText(bad), '');
  assert.equal(runsText(3, 0.02, 45), 'Your run waits · 3 runs as now, 3 with 0.02 · about 45 s');
  assert.equal(runsText(4, null, 90), 'Your run waits · 4 runs as now, 4 with this value · about 2 min');
  assert.equal(runsText(3, f32(0.3)), 'Your run waits · 3 runs as now, 3 with 0.3'); assert.equal(runsText(null, 0.02, 45), '');
  // What Run still needs: a cell, a number within the setting's bounds (far drift 0.1 to 0.3, the others Live knobs')
  // that is not the run's value now; far drift only while drift is on. Nothing about the run's or the check's state.
  const said = problems => problems.map(p => [p.field, p.label ? `${p.label}: ${p.message}` : p.message]);
  const ok = { label: 7, text: '0.02', flag: 'cell-compactness', setting: SETTINGS['cell-compactness'] };
  assert.deepEqual(testProblems(ok), []);
  assert.deepEqual(said(testProblems({ ...ok, label: 0 })), [[null, C.testNoCell]]);
  for (const text of ['', '  ', 'abc', undefined]) assert.deepEqual(said(testProblems({ ...ok, text })), [[TEST_FIELDS.value, 'Value to test: type a number.']]);
  assert.deepEqual(said(testProblems({ ...ok, text: '20' })), [[TEST_FIELDS.value, 'Value to test: use a number from 0.001 to 10.']]);
  assert.deepEqual(said(testProblems({ ...ok, text: '0.04' })), [[TEST_FIELDS.value, "Value to test: that is your run's value now: choose another."]],
    'the run\'s own value, as proseg reports it (0.0399999991)');
  const drift = { label: 7, flag: 'diffusion-probability', setting: SETTINGS['diffusion-probability'] };
  assert.deepEqual(testProblems({ ...drift, text: '0.3' }), []); assert.deepEqual(testProblems({ ...drift, text: '0.1' }), []);
  for (const text of ['0.05', '0.31', '1']) assert.deepEqual(said(testProblems({ ...drift, text })), [[TEST_FIELDS.value, 'Value to test: use a number from 0.1 to 0.3.']], text);
  assert.deepEqual(said(testProblems({ ...drift, text: '0.2' })), [[TEST_FIELDS.value, "Value to test: that is your run's value now: choose another."]]);
  assert.deepEqual(said(testProblems({ ...drift, text: '0.3', driftOff: true })), [[null, 'Far drift does nothing while drift is switched off in this run.']]);
  assert.deepEqual(testProblems({ ...ok, driftOff: true }), [], 'drift off is about far drift and the short drift distance only');
  const near = { label: 7, flag: 'diffusion-sigma-near', setting: { value: 1, min: 0.01, max: 100 } };
  assert.deepEqual(said(testProblems({ ...near, text: '2', driftOff: true })), [[null, C.testNearDriftOff]]);
  // A whole number, one of the values offered, the switch's 1 or 0, and no depth layers where the sample has no depth.
  const layers = { label: 7, flag: 'voxel-layers', setting: { value: 4, min: 1, max: 94 } };
  assert.deepEqual(testProblems({ ...layers, text: '8' }), []);
  for (const text of ['8.5', '95', '0']) assert.deepEqual(said(testProblems({ ...layers, text })), [[TEST_FIELDS.value, 'Value to test: use a whole number from 1 to 94.']], text);
  assert.deepEqual(said(testProblems({ ...layers, text: '8', flat: true })), [[null, C.testFlat]]);
  const grid = { label: 7, flag: 'burnin-voxel-size', setting: { value: 2 }, choices: [1, 3, 4] };
  assert.deepEqual(testProblems({ ...grid, text: '3' }), []);
  assert.deepEqual(said(testProblems({ ...grid, text: '1.5' })), [[TEST_FIELDS.value, `Value to test: ${C.testChoose}`]]);
  assert.deepEqual(said(testProblems({ ...grid, text: '' })), [[TEST_FIELDS.value, `Value to test: ${C.testChoose}`]]);
  const drifting = { label: 7, flag: 'no-diffusion', setting: { value: 0 } };
  assert.deepEqual(testProblems({ ...drifting, text: '1' }), []);
  assert.deepEqual(said(testProblems({ ...drifting, text: '0' })), [[TEST_FIELDS.value, "Value to test: that is your run's value now: choose another."]]);
  assert.deepEqual(said(testProblems({ ...drifting, text: '2' })), [[TEST_FIELDS.value, `Value to test: ${C.testChoose}`]]);
  // Shares as whole percentages, one decimal where two would read the same; a half rounds up as written.
  assert.deepEqual(percentPair(0.57, 0.52), ['57', '52']); assert.deepEqual(percentPair(0.665, 0.67), ['66.5', '67.0']);
  assert.deepEqual(percentPair(0.565, 0.5), ['57', '50']); assert.equal(tenths(0.565), 56.5); assert.equal(percent(0.565), 57);
  assert.deepEqual(percentPair(0.6, 0.6), ['60', '60'], 'equal shares: the same number'); assert.deepEqual(percentPair(null, 0.5), [C.missing, '50']);
  // Where they went: whole percentages adding up to 100 (largest remainder), each within a point of its share.
  assert.deepEqual(wholePercents([0.57, 0.06, 0.12, 0.25]), [57, 6, 12, 25]);
  assert.deepEqual(wholePercents([1 / 3, 1 / 3, 1 / 3, 0]), [34, 33, 33, 0], 'a tie: the first gets the point');
  assert.deepEqual(wholePercents([0.335, 0.335, 0.33, 0]), [34, 33, 33, 0]);
  assert.equal(wholePercents([0, 0, null, NaN]), null);
  for (const text of [poolLine({ cells: 2, suspects: 9, need: 160 }).text, tip, areaText(SQUARE), areaText(REGION), areaText({ rule: 'crop', box: [0, 0, 500, 180] }),
    runsText(3, 0.02, 45)]) {
    assert.deepEqual(bannedWords(text), [], text); assert.ok(clashFree(text), text);
  }
});

test('the reasons: the page\'s words and kinds are the engine\'s, code by code in its order, with their numbers; an unknown code has its kind\'s words', () => {
  assert.equal(REASONS.length, 39);
  assert.deepEqual(Object.keys(TEST_REASON_WORDS), REASONS.map(e => e.code), 'every code, in the engine\'s order');
  assert.deepEqual(Object.entries(TEST_REASON_KINDS), REASONS.map(e => [e.code, e.kind]), 'each code\'s kind, in the same order');
  // What still refuses (nothing to test, or the test can't run: a run without proseg's own count among them), what the app
  // does or waits for first, and the rest: the test runs, and its answer says where it was not measured or how its
  // suspects were taken. A cell that does not look mixed but glows is tested (low_share), one where nothing glows on
  // either count is not; one whose dots glow only on the count proseg saves is tested on those (saved_suspects, F1).
  assert.deepEqual(OF_KIND('refuse').map(e => e.code), ['ending', 'ended', 'no_counts', 'old_proseg', 'check_unavailable', 'busy', 'not_judged',
    'no_suspects', 'same_type', 'no_primary', 'no_controls']);
  assert.deepEqual(OF_KIND('wait').map(e => e.code), ['not_at_end', 'check_off', 'check_wait', 'check_scoring', 'details', 'other_run']);
  assert.deepEqual(OF_KIND('caveat').map(e => e.code), ['past_end', 'approx_targets', 'targets_now', 'whole_sample', 'tiled_box', 'schedule', 'nuclei_start',
    'lasso', 'area_narrow', 'square_not_measured', 'region_small', 'settings', 'count_basis', 'edge', 'saved_suspects', 'low_share', 'weak_suspects',
    'pattern_none', 'pattern_neighbour', 'pattern_other', 'pattern_mixed', 'too_few']);
  assert.equal(TEST_REASON_WORDS.saved_suspects, "Suspects taken from proseg's count as saved: its averaged count gives none that can be tested");
  for (const e of REASONS) {
    assert.equal(TEST_REASON_WORDS[e.code], e.words.replace(/\{(\w+):g\}/g, '{$1}'), e.code);
    assert.equal(testReasonText({ code: e.code, ...e.data }), e.shown, e.code);
    assert.deepEqual(bannedWords(e.shown), [], e.code); assert.ok(clashFree(e.shown), e.code);
    // A caveat says what was not measured, how the suspects were taken (approximately, where the run is now, from the
    // count as saved, weak, from a cell that does not look mixed), or that the test's numbers are for more suspects than
    // it has (too_few); a wait what starts it (or what the app does first); a refusal neither.
    if (e.kind === 'caveat') assert.match(e.shown, /^(Not measured yet (for|with) |Suspects (placed|taken) |Its suspects are weak|Doesn't look mixed overall|Its numbers are for tests with \d+ suspects or more: )/, e.code);
    else if (e.kind === 'wait') assert.match(e.shown, /^(Starts once|Switches .* on first$)/, e.code);
    else assert.doesNotMatch(e.shown, /not measured|starts once|first/i, e.code);
  }
  assert.equal(testReasonText({ code: 'region_small', min_um: 300.5 }), 'Not measured yet for regions under 300.5 µm');
  assert.equal(testReasonText({ code: 'too_few', have: 121, need: 160 }), 'Its numbers are for tests with 160 suspects or more: 121 here');
  assert.equal(testReasonText({ code: 'too_few' }), 'Its numbers are for tests with — suspects or more: — here', 'never braces');
  // A code the page doesn't know (a newer engine's): its kind's own words, never the engine's.
  assert.deepEqual([C.testReasonUnknown, C.testWaitUnknown, C.testCaveatUnknown], ["The test can't run here", 'Starts as soon as it can', 'Not measured yet here']);
  assert.equal(testReasonText({ code: 'a_new_reason', words: 'the engine\'s own words' }), C.testReasonUnknown);
  assert.equal(testReasonText({ code: 'a_new_reason' }, C.testCaveatUnknown), C.testCaveatUnknown);
  assert.equal(testReasonText(null), C.testReasonUnknown); assert.equal(testReasonText('ending'), REASONS[0].shown);
  for (const text of [C.testReasonUnknown, C.testWaitUnknown, C.testCaveatUnknown]) { assert.deepEqual(bannedWords(text), [], text); assert.ok(clashFree(text), text); }
  // A list as the engine gives it: each code once, in its order, in the page's words (never the engine's); waits and
  // caveats with their own words for a code the page doesn't know; their codes for the hooks.
  const listed = testReasons([{ code: 'ended', words: 'x' }, { code: 'no_suspects' }, { code: 'ended' }, { code: 7 }, null]);
  assert.deepEqual(listed, [{ code: 'ended', text: 'This run ended before its last iteration' }, { code: 'no_suspects', text: 'No suspects to test in this cell: none glows, or none could be placed' }]);
  assert.deepEqual(testWaits([{ code: 'not_at_end' }, { code: 'a_new_wait' }]), [{ code: 'not_at_end', text: 'Starts once your run reaches its last iteration' },
    { code: 'a_new_wait', text: C.testWaitUnknown }]);
  assert.deepEqual(testCaveats([{ code: 'too_few', have: 74, need: 160 }, { code: 'a_new_caveat' }]),
    [{ code: 'too_few', text: 'Its numbers are for tests with 160 suspects or more: 74 here' }, { code: 'a_new_caveat', text: C.testCaveatUnknown }]);
  assert.deepEqual(reasonCodes([{ code: 'lasso' }, { code: 'edge' }, { code: 'lasso' }, 'x', null]), ['lasso', 'edge']);
  assert.deepEqual([reasonCodes(null), reasonCodes(undefined), testWaits('x'), testCaveats(null)], [[], [], [], []]);
});

test('the refusals\' groups: what passes by itself first, what leaves nothing to test or can\'t run; a press says one line by them', () => {
  assert.deepEqual(TEST_REASON_GROUPS, ['first', 'cant'], 'nothing is refused for not being measured: no group for it');
  const refusals = OF_KIND('refuse').map(e => e.code), of = group => refusals.filter(code => testReasonGroup(code) === group);
  assert.deepEqual(of('first'), ['busy']);
  assert.deepEqual(of('cant'), ['ending', 'ended', 'no_counts', 'old_proseg', 'check_unavailable', 'not_judged', 'no_suspects', 'same_type', 'no_primary',
    'no_controls']);
  // Were a wait ever listed it would pass by itself too; anything else, and a code the page doesn't know, can't be tested.
  for (const e of OF_KIND('wait')) assert.equal(testReasonGroup(e.code), 'first', e.code);
  for (const code of ['too_few', 'a_new_reason', '', null, undefined, 7, 'toString', '__proto__']) assert.equal(testReasonGroup(code), 'cant', String(code));
  // A press of Run while they are listed: a reason about the run or its sample (no cell of it can be tested: the run is
  // named, never the cell); else one that leaves this cell nothing to test (a code the page doesn't know too); else wait
  // for the test of this run on its way. Never "Check these first:".
  assert.deepEqual(TEST_RUN_REFUSALS, ['ending', 'ended', 'check_unavailable', 'no_counts', 'old_proseg']);
  assert.ok(TEST_RUN_REFUSALS.every(code => refusals.includes(code)), 'refusals, each');
  const line = (...codes) => testNotHere(testReasons(codes.map(code => ({ code }))));
  assert.equal(line('busy'), C.testNotHereFirst);
  assert.deepEqual([line('ending'), line('ended'), line('check_unavailable'), line('no_counts'), line('old_proseg'), line('busy', 'ending'),
    line('ended', 'no_suspects')], Array(7).fill(C.testNotHereRun));
  assert.deepEqual([line('not_judged'), line('no_suspects'), line('same_type'), line('no_primary'), line('busy', 'no_suspects'),
    line('a_new_reason')], Array(6).fill(C.testNotHereCant));
  assert.deepEqual([line(), testNotHere(null)], ['', '']);
  assert.deepEqual([C.testNotHereRun, C.testNotHereCant, C.testNotHereFirst], ["This run can't be tested.", "This cell can't be tested.", 'Wait until it ends.']);
  assert.ok(!('testNotHereUnmeasured' in C), 'never "not measured yet here: nothing to change"');
  for (const text of [C.testNotHereRun, C.testNotHereCant, C.testNotHereFirst]) {
    assert.deepEqual(bannedWords(text), [], text); assert.ok(clashFree(text), text); assert.doesNotMatch(text, /check these/i, text);
  }
});

test('the entry is shown for a selected cell of a proseg run whatever the run\'s state or the mixing check\'s; their reasons are the form\'s', t => {
  const e = page(t), p = e.panel;
  assert.equal(p.body.childNodes[0], p.cellRow); assert.equal(p.body.childNodes[1], e.box, 'right under the row');
  assert.equal(e.box.hidden, false); assert.equal(e.box.dataset.state, 'entry');
  const open = e.find('test-open');
  assert.equal(open.textContent, C.testOpen); assert.equal(open.title, C.testOpenTooltip); assert.equal(open.disabled, false);
  assert.deepEqual(e.models, [], 'nothing handed to the corner'); assert.deepEqual(p.testCornerModel, { open: false });
  assert.deepEqual(e.previews, [], 'nothing asked before the form opens');
  for (const run of ['preparing', 'running', 'paused', 'finishing', 'done', 'stopped', 'failed']) {
    p.setTestRun({ id: 'r1', state: run, ...PAIRED }); assert.equal(e.box.dataset.state, 'entry', run); assert.equal(p.testable(), true, run);
  }
  for (const s of [{ enabled: false, state: 'off' }, { state: 'unavailable' }, { state: 'failed' }, { state: 'learning' }, { state: 'waiting' }]) {
    e.state(s); assert.equal(e.box.dataset.state, 'entry', JSON.stringify(s));
  }
  e.state();
  // A cell given (Go deeper's own) is testable as the selected one is; no cell, a run that is not proseg's, none: not.
  assert.equal(p.testable(12), true); assert.equal(p.testable(0), false); assert.equal(p.testable('7'), false);
  p.setTestRun({ id: 'r1', state: 'paused', kind: 'stain' }); assert.equal(e.box.hidden, true); assert.equal(p.testable(7), false);
  p.setTestRun({ id: 'r1', state: 'paused', ...PAIRED }); p.setCell(null);
  assert.equal(e.box.hidden, true, 'no cell selected'); assert.equal(p.testShown, null); assert.equal(p.testable(), false);
  p.setTestRun(null); assert.equal(p.testable(7), false);
  // A page that wires no test offers none.
  const bare = page(t, { wire: false });
  assert.equal(bare.box.hidden, true); assert.equal(bare.find('test-open'), undefined); assert.equal(bare.panel.testable(7), false);
});

test('the form: three settings in one order for every start, nothing prefilled, the run\'s value beside, far drift not tried yet and its range', async t => {
  for (const [record, start] of [[PAIRED, 'paired'], [CELLS, 'cells']]) {
    const e = page(t, { record });
    await e.open();
    assert.equal(e.box.dataset.state, 'form'); assert.equal(e.models.at(-1).open, true);
    const m = e.model();
    assert.deepEqual([m.open, m.phase, m.run, m.cell, m.start], [true, 'form', 'r1', { label: 7, name: 'ffaaaaaa-7' }, start]);
    assert.deepEqual(m.form.settings.map(s => [s.flag, s.name, s.now, s.nowText, s.tried, s.note, s.tip, s.min, s.max]), [
      ['cell-compactness', 'Cell compactness', f32(0.04), 'now 0.04', true, '', '', 0.001, 10],
      ['prior-weight', C.testHold, 1, 'now 1', true, '', '', 0, 1000],
      ['diffusion-probability', C.testFarDrift, f32(0.2), 'now 0.2', false, C.testNotTriedNote, C.testFarDriftTip, 0.1, 0.3],
      ['burnin-voxel-size', 'Starting grid (µm)', null, '', false, C.testNotTriedNote, copy('testGridTip', { v: '1' }), null, null],
      ['ab-nihlo-bubble-prob', 'Background gaps', null, '', false, C.testNotTriedNote, C.testGapsTip, null, null],
      ['no-diffusion', 'Drift (on or off)', null, '', false, C.testNotTriedNote, C.testDriftTip, null, null],
      ['diffusion-sigma-near', 'Short drift distance (µm)', null, '', false, C.testNotTriedNote, C.testNearDriftTip, null, null],
      ['density-bins', 'Crowding levels', null, '', false, C.testNotTriedNote, C.testCrowdingTip, null, null],
      ['voxel-layers', 'Depth layers', null, '', false, C.testNotTriedNote, C.testLayersTip, null, null]], start);
    assert.deepEqual([m.form.flag, m.form.value, m.form.problems, m.form.field], ['cell-compactness', '', [], null], 'nothing prefilled');
    assert.equal(m.form.line, 'Setting: Cell compactness 0.04 → — (lower = rounder cells)');
    // The form's only notes are far drift's; how the runs are made is the runs line's tooltip (the How line), never what
    // they may show.
    const notes = m.form.settings.flatMap(s => [s.note, s.tip]).filter(Boolean);
    assert.deepEqual(notes, [C.testNotTriedNote, C.testFarDriftTip, C.testNotTriedNote, copy('testGridTip', { v: '1' }), C.testNotTriedNote, C.testGapsTip,
      C.testNotTriedNote, C.testDriftTip, C.testNotTriedNote, C.testNearDriftTip, C.testNotTriedNote, C.testCrowdingTip, C.testNotTriedNote, C.testLayersTip]);
    assert.equal(m.form.how, `${C.testHow} ${C.testHowHold}`, 'how the runs are made, and that their hold is as strong as the run\'s (P4)');
    for (const flag of TEST_FLAGS) { await e.act('setting', flag); assert.deepEqual([e.form().flag, e.form().value], [flag, ''], `${start}: ${flag} starts empty`); }
    await e.act('setting', 'diffusion-probability'); await e.act('value', '0.25'); assert.equal(e.form().value, '0.25');
    assert.equal(e.form().line, 'Setting: Share of far-drifted transcripts 0.2 → 0.25', 'far drift by its Live knobs name, without the banned hint');
    // A setting proseg reads at its start, which Live knobs does not hold: the deck's own Setting line, its unit said.
    await e.act('setting', 'voxel-layers'); await e.act('value', '8'); assert.equal(e.form().line, 'Setting: Depth layers — → 8');
    await e.act('setting', 'cell-compactness'); assert.equal(e.form().value, '', 'another setting: its field empty again');
    await e.act('setting', 'dispersion'); assert.equal(e.form().flag, 'cell-compactness', 'only the settings a test offers');
    assert.deepEqual(banned(e), []);
    // The panel keeps one line while the test is open.
    assert.equal(e.find('test-open'), undefined); assert.equal(e.status().textContent, C.testStatusOpen); assert.equal(e.status().dataset.phase, 'open');
    assert.deepEqual(e.lines(), [C.testStatusOpen]); assert.equal(e.status().title, C.testStatusTooltip);
  }
});

test('the pool is asked for once per run, cell and end of the run, "Finding…" only after 250 ms, a pool being worked out again with wait 5, every 2 s while a wait is listed', async t => {
  const e = page(t), p = e.panel;
  let release;
  e.reply.pool = () => new Promise(resolve => { release = resolve; });
  await e.open();
  assert.deepEqual(e.previews, [[7, 0, true]], 'the cell\'s label (main.js asks for label - 1), no wait, a signal to give up');
  assert.equal(e.form().pool, null, 'nothing said yet');
  await e.advance(TEST_POOL_SHOW_MS - 1); assert.equal(e.form().pool, null);
  await e.advance(1);
  assert.deepEqual(e.form().pool, { state: 'asking', text: C.testPoolAsking, cells: null, suspects: null, need: null });
  release(POOL()); await settle(); await settle();
  let f = e.form();
  assert.deepEqual(f.pool, { state: 'reached', text: 'Nearby cells with the same pattern: 23 cells, 187 suspects',
    tooltip: 'The test was measured with 160 suspects or more. With fewer it runs all the same, and its answer says so.', cells: 23, suspects: 187, need: 160 });
  assert.deepEqual(f.area, { rule: 'square', text: 'Area: a 500 × 500 µm square around this cell' });
  assert.equal(f.runs, 'Your run waits · 3 runs as now, 3 with this value · about 45 s');
  assert.deepEqual([f.reasons, f.waits, f.caveats], [[], [], []]);
  await e.act('value', '0.02'); assert.equal(e.form().runs, 'Your run waits · 3 runs as now, 3 with 0.02 · about 45 s');
  // Typing, another setting, the same run again, a push: never asked again.
  e.reply.pool = () => POOL();
  await e.act('setting', 'prior-weight'); e.run(); e.run({ qc_test: null }); e.state(); await e.advance(5000);
  assert.equal(e.previews.length, 1, 'once');
  // The run's end moved (sampled on and paused at another end), another cell, the mixing check switched on: asked again.
  e.run({ schedule_done: { iter: 900, seq: 901 } }); await settle(); assert.equal(e.previews.length, 2);
  p.setCell({ label: 9, name: 'ffaaaaaa-9', color: '#456', line: '' }); await settle(); assert.deepEqual(e.previews.at(-1), [9, 0, true]);
  e.state({ enabled: false }); await settle(); e.state(); await settle(); assert.equal(e.previews.length, 5);
  // A pool still being worked out: its cheap fields at once, asked again waiting up to 5 s (never in a loop).
  const working = POOL({ state: 'working', offer: false, reasons: [], pool: null, area: REGION, runs_per_side: 4, estimate_s: 50 });
  let asks = 0;
  e.reply.pool = (label, options) => { asks++; return asks < 3 ? working : POOL(); };
  p.setCell({ label: 11, name: 'ffaaaaaa-11', color: '#456', line: '' }); await settle(); await settle();
  assert.deepEqual(e.previews.at(-1), [11, 0, true]);
  assert.deepEqual([e.form().pool, e.form().area?.text, e.form().runs], [null, 'Area: your whole region, 300 × 300 µm', 'Your run waits · 4 runs as now, 4 with this value · about 50 s']);
  await e.advance(TEST_POOL_SHOW_MS);
  assert.deepEqual(e.previews.at(-1), [11, TEST_POOL_WAIT_S, true]); assert.equal(e.form().pool.state, 'asking', '"Finding…" while it is worked out');
  await e.advance(TEST_POOL_SHOW_MS); assert.equal(asks, 3); assert.equal(e.form().pool.state, 'reached');
  // A wait that passes by itself (the check scoring the last iteration): in Run's tooltip (the form's waits), and asked
  // about again 2 s later; never a refusal.
  const scoring = OFFERED([{ code: 'check_scoring', wait: true, words: 'x' }]);
  let n = 0;
  e.reply.pool = () => (++n === 1 ? scoring : POOL());
  p.setCell({ label: 12, name: 'ffaaaaaa-12', color: '#456', line: '' }); await settle(); await settle();
  assert.deepEqual([e.form().reasons, e.form().waits], [[], [{ code: 'check_scoring', text: "Starts once the mixing check has scored your run's last iteration" }]]);
  const before = e.previews.length;
  await e.advance(TEST_POOL_AGAIN_MS - 1); assert.equal(e.previews.length, before);
  await e.advance(1); assert.equal(e.previews.length, before + 1); assert.deepEqual(e.form().waits, [], 'scored now: nothing to wait for');
  await e.advance(10000); assert.equal(e.previews.length, before + 1, 'asked about once: nothing loops');
  // Any wait is asked about again every 2 s while it is listed (the check learning, Go deeper's details, another run: the
  // page sees none of them pass), so Run's tooltip says what holds now and never one that passed; once none is listed,
  // nothing more is asked.
  const waits = [OFFERED(['check_wait', 'details']), OFFERED(['check_wait', 'details']), OFFERED(['details']), OFFERED(['other_run']), POOL()];
  let w = 0;
  e.reply.pool = () => waits[Math.min(w++, waits.length - 1)];
  p.setCell({ label: 15, name: 'ffaaaaaa-15', color: '#456', line: '' }); await settle(); await settle();
  const listed = () => e.form().waits.map(x => x.code), first = e.previews.length;
  assert.deepEqual(listed(), ['check_wait', 'details']);
  await e.advance(TEST_POOL_AGAIN_MS - 1); assert.equal(e.previews.length, first);
  await e.advance(1); assert.deepEqual([e.previews.length, e.previews.at(-1), listed()], [first + 1, [15, 0, true], ['check_wait', 'details']], 'still learning');
  await e.advance(TEST_POOL_AGAIN_MS); assert.deepEqual([e.previews.length, listed()], [first + 2, ['details']], 'learned: its wait gone from the tooltip');
  await e.advance(TEST_POOL_AGAIN_MS); assert.deepEqual([e.previews.length, listed()], [first + 3, ['other_run']]);
  await e.advance(TEST_POOL_AGAIN_MS); assert.deepEqual([e.previews.length, listed(), e.form().pool.state], [first + 4, [], 'reached'], 'nothing holds');
  await e.advance(10000); assert.equal(e.previews.length, first + 4, 'nothing listed: nothing loops');
  // One the engine could not work out (or an ask that failed): said, and Run asks for the start anyway.
  e.reply.pool = () => POOL({ state: 'failed', message: 'x' });
  p.setCell({ label: 13, name: 'ffaaaaaa-13', color: '#456', line: '' }); await settle(); await settle();
  assert.deepEqual(e.form().pool, { state: 'failed', text: C.testPoolFailed, cells: null, suspects: null, need: null });
  e.reply.pool = () => Promise.reject(Object.assign(new Error('Unknown cell'), { status: 404 }));
  p.setCell({ label: 14, name: 'ffaaaaaa-14', color: '#456', line: '' }); await settle(); await settle();
  assert.equal(e.form().pool.state, 'failed');
  // Closed: nothing more asked, nothing left scheduled for it.
  await e.act('close'); assert.deepEqual(e.model(), { open: false }); assert.equal(e.tasks.size, 0);
  assert.deepEqual(banned(e), []);
});

test('the form lists the refusals alone, in the engine\'s order and the page\'s words; what a press would wait for and where it was not measured go to Run\'s tooltip', async t => {
  // A reply with every reason of the shared fixture, each in its list (the engine's order kept in each).
  const e = page(t, { run: 'running', record: { ...PAIRED, schedule_done: null } });
  e.reply.pool = () => REFUSED(OF_KIND('refuse').map(GIVEN), { area: null, waits: OF_KIND('wait').map(GIVEN), caveats: OF_KIND('caveat').map(GIVEN) });
  await e.open(); await settle();
  const f = e.form(), words = kind => OF_KIND(kind).map(r => ({ code: r.code, text: r.shown }));
  assert.deepEqual([f.reasons, f.waits, f.caveats], [words('refuse'), words('wait'), words('caveat')]);
  assert.deepEqual([f.pool, f.area], [null, null], 'refused before the pool: no pool line');
  assert.deepEqual([e.panel.testShown.reasons, e.panel.testShown.waits, e.panel.testShown.caveats], ['refuse', 'wait', 'caveat'].map(kind => OF_KIND(kind).map(r => r.code)),
    'the harness reads the codes');
  assert.ok(!modelWords(e.model()).some(w => w.includes("engine's own")), 'the engine\'s own words never shown');
  assert.deepEqual(banned(e), []);
  // A start from cells only on a large region, the run not at its end yet: offered; a press would wait for the run's
  // end, and the test was not measured there (both in Run's tooltip), its 4 runs a side.
  const c = page(t, { run: 'running', record: { ...CELLS, schedule_done: null } });
  c.reply.pool = () => OFFERED(['not_at_end'], ['square_not_measured'], { start: 'cells', runs_per_side: 4, area: null, estimate_s: 50,
    row: { start: 'cells', situation: 'as_is' } });
  await c.open(); await settle();
  assert.deepEqual([c.form().reasons, c.form().waits.map(r => r.text), c.form().caveats.map(r => r.text)],
    [[], ['Starts once your run reaches its last iteration'], ['Not measured yet for cells-only starts on regions over 500 µm']]);
  assert.equal(c.form().runs, 'Your run waits · 4 runs as now, 4 with this value · about 50 s');
  assert.deepEqual([c.panel.testShown.reasons, c.panel.testShown.waits, c.panel.testShown.caveats], [[], ['not_at_end'], ['square_not_measured']]);
  // A reply still being worked out says the waits and caveats known so far.
  const w = page(t);
  let asks = 0;
  w.reply.pool = () => (++asks === 1 ? OFFERED(['check_wait'], ['lasso'], { state: 'working', pool: null }) : new Promise(() => {}));
  await w.open(); await settle();
  assert.deepEqual([w.form().waits.map(r => r.code), w.form().caveats.map(r => r.code)], [['check_wait'], ['lasso']]);
});

test('Run waits for the pool on its way, asks nothing where the engine refused, and posts {cell, flag, value} only', async t => {
  const e = page(t), p = e.panel;
  let release;
  e.reply.pool = () => new Promise(resolve => { release = resolve; });
  await e.open();
  // What the page itself sees comes first: a value, within its bounds, not the run's own; asked of nothing.
  const check = async (text, expected, field = 'value') => {
    await e.act('value', text); await e.act('run');
    assert.deepEqual(e.started, [], 'nothing asked');
    assert.deepEqual(e.form().problems, [expected]); assert.equal(e.form().field, field); assert.equal(e.model().phase, 'form');
  };
  await check('', 'Value to test: type a number.');
  await check('0.04', "Value to test: that is your run's value now: choose another.");
  await check('11', 'Value to test: use a number from 0.001 to 10.');
  await e.act('value', '0.02'); assert.deepEqual(e.form().problems, [], 'what it said goes as the value changes');
  // The pool is on its way: Run waits for it (starting), then asks.
  const pressed = e.act('run');
  await settle();
  assert.deepEqual(e.started, []); assert.deepEqual([e.model().phase, e.model().busy, e.model().busyText], ['starting', 'run', C.testStarting]);
  assert.deepEqual([e.model().test.flag, e.model().test.value, e.model().test.valueText], ['cell-compactness', 0.02, '0.02']);
  await e.act('run'); assert.equal(e.model().message, EXPLAIN_TEXT.busy, 'a second press asks nothing');
  release(POOL()); await pressed; await settle();
  assert.deepEqual(e.started, [[7, 'cell-compactness', 0.02]], 'the label (main.js sends label - 1), the setting and the value as a number, nothing else');
  assert.equal(e.box.dataset.state, 'preparing');
  // A pool the engine refused: Run asks for it again first (what holds now), and asks nothing while it is refused.
  const r = page(t);
  r.reply.pool = () => REFUSED([{ code: 'busy', why: 'test', words: 'x' }]);
  await r.open(); await settle(); await r.act('value', '0.02');
  const asked = r.previews.length;
  await r.act('run'); await settle();
  assert.equal(r.previews.length, asked + 1, 'asked again on the press'); assert.deepEqual(r.started, [], 'never posted while refused');
  assert.deepEqual([r.form().problems, r.form().notHere, r.form().reasons.map(x => x.code), r.model().phase], [[], C.testNotHereFirst, ['busy'], 'form'],
    'one line, never a problem to check');
  // Asked again, it holds no more (the test there ended meanwhile): posted.
  r.reply.pool = () => POOL();
  await r.act('run'); await settle();
  assert.deepEqual(r.started, [[7, 'cell-compactness', 0.02]]);
  // Each reason that still refuses, one at a time: nothing to test in this cell, or the test can't run on this run or
  // sample (the run named then); their own line, nothing posted. busy (a test of this run on its way): wait until it ends.
  for (const e2 of OF_KIND('refuse')) {
    const n = page(t);
    n.reply.pool = () => REFUSED([GIVEN(e2)]);
    await n.open(); await settle(); await n.act('value', '0.02'); await n.act('run'); await settle();
    const said = e2.code === 'busy' ? C.testNotHereFirst : TEST_RUN_REFUSALS.includes(e2.code) ? C.testNotHereRun : C.testNotHereCant;
    assert.deepEqual([n.started, n.form().problems, n.form().notHere, n.form().reasons.map(x => x.code)], [[], [], said, [e2.code]], e2.code);
    await n.act('value', '0.03'); assert.equal(n.form().notHere, '', 'what it said goes as the value changes');
  }
  // Asked again, the pool is being worked out once more ("working": never a refusal): Run waits for it, then asks.
  const wk = page(t);
  let k = 0;
  wk.reply.pool = () => [REFUSED(['ending']), POOL({ state: 'working', offer: false, reasons: [], pool: null }), POOL()][Math.min(k++, 2)];
  await wk.open(); await settle(); await wk.act('value', '0.02');
  const pressing = wk.act('run'); await settle(); await settle();
  assert.deepEqual([wk.started, wk.form().problems], [[], []], 'working: Run waits, nothing said');
  await wk.advance(TEST_POOL_SHOW_MS); await pressing; await settle();
  assert.deepEqual(wk.started, [[7, 'cell-compactness', 0.02]]); assert.deepEqual(wk.previews.map(p => p[1]), [0, 0, TEST_POOL_WAIT_S]);
  // A pool the engine could not count: Run asks for the start anyway (the engine counts it again).
  const f = page(t);
  f.reply.pool = () => POOL({ state: 'failed' });
  await f.open(); await settle(); await f.act('setting', 'diffusion-probability'); await f.act('value', '0.3'); await f.act('run'); await settle();
  assert.deepEqual(f.started, [[7, 'diffusion-probability', 0.3]]);
  // Far drift with drift off in the run: said, nothing asked.
  const d = page(t, { record: { ...PAIRED, options: { 'no-diffusion': true } } });
  await d.open(); await settle(); await d.act('setting', 'diffusion-probability'); await d.act('value', '0.3'); await d.act('run');
  assert.deepEqual([d.started, d.form().problems], [[], [C.testDriftOff]]);
  // A page whose pool never answers: Run waits TEST_POOL_RUN_WAIT_MS at most, then asks (the engine works it out).
  const w = page(t);
  w.reply.pool = () => new Promise(() => {});
  await w.open(); await w.act('value', '0.02'); const waiting = w.act('run'); await settle();
  await w.advance(TEST_POOL_RUN_WAIT_MS - 1); assert.deepEqual(w.started, []);
  await w.advance(1); await waiting; assert.deepEqual(w.started, [[7, 'cell-compactness', 0.02]]);
  // A page without the preview (an older page's options): Run asks at once.
  const o = page(t, { preview: false });
  await o.open(); await o.act('value', '0.02'); await o.act('run'); await settle();
  assert.deepEqual(o.started, [[7, 'cell-compactness', 0.02]]);
});

test('a test of the hold is checked against its value at the frame tested, the pool reply\'s: a hold set live since then is taken, the one it had then refused in its own words (F5)', async t => {
  // The engine tests the hold against the run's at F (its last iteration: the pool reply's `hold`, {value, ...} at F),
  // never its value now: here set live to 1.5 since F, where it was 1.
  const said = problems => problems.map(p => [p.field, p.label ? `${p.label}: ${p.message}` : p.message]);
  const live = { ...SETTINGS['prior-weight'], value: 1.5 }, atF = { value: 1, effective: 7.62241, scale: 7.62241, seq: 451, given: null };
  const hold = { label: 7, flag: 'prior-weight', setting: live };
  const usedThen = "Value to test: that was your run's value at its last iteration: choose another.";
  assert.deepEqual(testProblems({ ...hold, text: '1.5', hold: atF }), [], 'its value now: the engine takes it');
  assert.deepEqual(said(testProblems({ ...hold, text: '1', hold: atF })), [[TEST_FIELDS.value, usedThen]]);
  // ... a hold that did not change since F: the value now's words.
  assert.deepEqual(said(testProblems({ ...hold, setting: SETTINGS['prior-weight'], text: '1', hold: atF })),
    [[TEST_FIELDS.value, "Value to test: that is your run's value now: choose another."]]);
  // ... before the pool reply says it, not yet (Run checks again once it came); a reply that never says it: the value now.
  assert.deepEqual(testProblems({ ...hold, text: '1.5', holdDue: true }), []);
  assert.deepEqual(said(testProblems({ ...hold, text: '1.5' })), [[TEST_FIELDS.value, "Value to test: that is your run's value now: choose another."]]);
  // Another setting is checked against its value now, whatever the hold.
  assert.deepEqual(said(testProblems({ label: 7, flag: 'cell-compactness', setting: SETTINGS['cell-compactness'], text: '0.04', hold: atF, holdDue: true })),
    [[TEST_FIELDS.value, "Value to test: that is your run's value now: choose another."]]);
  assert.equal(C.testValueSameEnd, "that was your run's value at its last iteration: choose another.");
  assert.deepEqual(bannedWords(C.testValueSameEnd), []); assert.ok(clashFree(C.testValueSameEnd));
  // In the form: the pool reply carries F's hold, the run's hold now is the one set live since.
  const settings = { ...SETTINGS, 'prior-weight': live };
  const e = page(t, { settings });
  e.reply.pool = () => POOL({ hold: atF });
  await e.open(); await settle(); await e.act('setting', 'prior-weight');
  await e.act('value', '1'); await e.act('run'); await settle();
  assert.deepEqual([e.started, e.form().problems, e.form().field], [[], [usedThen], 'value'], 'F\'s value: never posted');
  await e.act('value', '1.5'); await e.act('run'); await settle();
  assert.deepEqual(e.started, [[7, 'prior-weight', 1.5]], 'the value set live since F: posted');
  // A press before the pool reply came: Run waits for it, then checks the hold against it.
  const late = page(t, { settings });
  let release;
  late.reply.pool = () => new Promise(resolve => { release = resolve; });
  await late.open(); await late.act('setting', 'prior-weight'); await late.act('value', '1.5');
  const pressing = late.act('run'); await settle();
  assert.deepEqual([late.started, late.model().phase], [[], 'starting'], 'its value now is not refused before the reply says F\'s');
  release(POOL({ hold: atF })); await pressing; await settle();
  assert.deepEqual(late.started, [[7, 'prior-weight', 1.5]]);
  const lateF = page(t, { settings });
  lateF.reply.pool = () => new Promise(resolve => { release = resolve; });
  await lateF.open(); await lateF.act('setting', 'prior-weight'); await lateF.act('value', '1');
  const pressingF = lateF.act('run'); await settle();
  release(POOL({ hold: atF })); await pressingF; await settle();
  assert.deepEqual([lateF.started, lateF.form().problems, lateF.model().phase], [[], [usedThen], 'form'], 'F\'s value, once the reply said it');
});

test('before the run reached its last iteration a test of the hold is checked against its value now, as the engine checks it then, whatever the pool reply said when it was made (F5)', async t => {
  // A pool reply made before F says the hold as it was then, with no frame (seq null: engine qc_test.hold_tested); the
  // POST then checks the value now (no F yet: the run's value now). Here set live to 1.5 since a reply said 1.
  const said = problems => problems.map(p => [p.field, p.label ? `${p.label}: ${p.message}` : p.message]);
  const live = { ...SETTINGS['prior-weight'], value: 1.5 }, beforeF = { value: 1, effective: 7.62241, scale: 7.62241, seq: null };
  const nowWords = "Value to test: that is your run's value now: choose another.";
  assert.deepEqual([holdAtEnd(beforeF), holdAtEnd({ ...beforeF, seq: 451 }), holdAtEnd(null), holdAtEnd({ value: null, seq: 451 })], [null, 1, null, null]);
  const hold = { label: 7, flag: 'prior-weight', setting: live, hold: beforeF };
  assert.deepEqual(testProblems({ ...hold, text: '1' }), [], 'the value the reply said: the engine takes it');
  assert.deepEqual(said(testProblems({ ...hold, text: '1.5' })), [[TEST_FIELDS.value, nowWords]], 'the value now: refused, as the engine does');
  // In the form: the reply lists the wait for the run's last iteration, so it is asked again every TEST_POOL_AGAIN_MS;
  // the form checks the value now, also while it is asked again (no gap in the check).
  const settings = { ...SETTINGS, 'prior-weight': live }, record = { ...PAIRED, schedule_done: null };
  const reply = () => OFFERED(['not_at_end'], [], { seq: null, iter: null, hold: beforeF });
  const e = page(t, { settings, record });
  e.reply.pool = reply;
  await e.open(); await settle(); await e.act('setting', 'prior-weight');
  await e.act('value', '1.5'); await e.act('run'); await settle();
  assert.deepEqual([e.started, e.form().problems, e.form().field], [[], [nowWords], 'value']);
  let release;
  e.reply.pool = () => new Promise(resolve => { release = resolve; });
  await e.advance(TEST_POOL_AGAIN_MS);
  assert.equal(e.panel.testPool.asking, true, 'asked again, its answer on its way');
  await e.act('value', '1.5'); const pressing = e.act('run'); await settle();
  assert.deepEqual([e.started, e.form().problems, e.model().phase], [[], [nowWords], 'form'], 'refused at once while asked again');
  release(reply()); await pressing; await settle();
  assert.deepEqual(e.started, [], 'nor once its answer came');
  await e.act('value', '1'); await e.act('run'); await settle();
  assert.deepEqual(e.started, [[7, 'prior-weight', 1]], 'the value the reply said when it was made: posted');
});

test('a test of the hold set live since the run\'s last iteration: the form keeps "now", says in its tooltip what the test compares with, and its Setting line starts from that (F5)', async t => {
  // The test's runs as now are given the hold at F (the pool reply's `hold`), which is what the test compares with: one
  // set live since then is the value now, said so beside the field, the value at F in its tooltip.
  const live = { ...SETTINGS['prior-weight'], value: 1.5 }, atF = { value: 1, effective: 7.62241, scale: 7.62241, seq: 451, given: null };
  const tip = "The test compares your value with 1, your run's value at its last iteration.";
  assert.equal(copy('testNowEnd', { v: settingValue(1) }), tip);
  assert.deepEqual(bannedWords(tip), []); assert.ok(clashFree(tip));
  const entry = env => env.form().settings.find(s => s.flag === 'prior-weight');
  const e = page(t, { settings: { ...SETTINGS, 'prior-weight': live } });
  e.reply.pool = () => POOL({ hold: atF });
  await e.open(); await settle(); await e.act('setting', 'prior-weight'); await e.act('value', '2');
  assert.deepEqual([entry(e).now, entry(e).nowText, entry(e).nowTip, entry(e).from], [1.5, 'now 1.5', tip, 1]);
  assert.equal(e.form().line, 'Setting: Prior weight 1 → 2 (higher = the starting cells hold their transcripts harder)',
    'what the test compares, as the test itself says it once it runs (its `current`)');
  assert.deepEqual(e.form().settings.filter(s => s.flag !== 'prior-weight').map(s => [s.flag, s.nowTip, s.from]),
    [['cell-compactness', '', f32(0.04)], ['diffusion-probability', '', f32(0.2)], ...['burnin-voxel-size', 'ab-nihlo-bubble-prob', 'no-diffusion',
      'diffusion-sigma-near', 'density-bins', 'voxel-layers'].map(flag => [flag, '', null])], 'the other settings: their value now (none said yet: no Live knob, no reply)');
  assert.deepEqual(banned(e), []);
  // A hold unchanged since F, or a reply made before F (the engine then compares with the value now): "now" alone.
  for (const [settings, hold] of [[SETTINGS, atF], [{ ...SETTINGS, 'prior-weight': live }, { ...atF, seq: null }]]) {
    const other = page(t, { settings });
    other.reply.pool = () => POOL({ hold });
    await other.open(); await settle(); await other.act('setting', 'prior-weight'); await other.act('value', '2');
    const now = settings['prior-weight'].value;
    assert.deepEqual([entry(other).nowTip, entry(other).from, other.form().line],
      ['', now, `Setting: Prior weight ${now} → 2 (higher = the starting cells hold their transcripts harder)`], `the reply's hold at seq ${hold.seq}`);
  }
});

test('a refused start: a 409\'s reasons listed, one without them is "busy" and a test there is asked for; a value, a cell, the rest in plain words', async t => {
  const e = page(t), p = e.panel;
  await e.open(); await settle(); await e.act('value', '0.02');
  e.reply.start = () => Promise.reject(refusal(409, { detail: 'x', status: 409, reasons: [{ code: 'ended', words: 'x' }, { code: 'no_suspects', words: 'x' }] }));
  await e.act('run'); await settle();
  let m = e.model();
  assert.deepEqual([m.phase, m.busy, m.form.problems, m.form.notHere], ['form', null, [], C.testNotHereRun], 'the run ended: no cell of it can be tested');
  assert.deepEqual(m.form.reasons, [{ code: 'ended', text: 'This run ended before its last iteration' }, { code: 'no_suspects', text: 'No suspects to test in this cell: none glows, or none could be placed' }]);
  assert.deepEqual(p.testShown.reasons, ['ended', 'no_suspects']);
  assert.ok(!modelWords(m).some(w => w.includes("engine's own")), 'the engine\'s own words never shown');
  // Without reasons (a test of this run on its way: pressed in another window): the busy reason, and the test there is
  // asked for.
  e.reply.start = () => Promise.reject(refusal(409, { detail: 'A test of this run is on its way: wait for its answer, or cancel it', status: 409 }));
  e.reply.fetch = () => RECORD({ id: 'other', state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 5 }) });
  await e.act('run'); await settle();
  assert.deepEqual([e.form().reasons, e.form().notHere], [[{ code: 'busy', text: 'A test of this run is on its way' }], C.testNotHereFirst]);
  await e.advance(1000); assert.equal(e.box.dataset.state, 'running', 'the test on its way there is shown'); assert.equal(e.model().phase, 'running');
  // A value refused names the value; a cell gone; anything else.
  const r = page(t);
  await r.open(); await settle(); await r.act('value', '0.02');
  for (const [status, expected, field] of [[422, "Value to test: this value can't be used for this setting.", 'value'], [404, C.testCellMissing, null],
    [500, C.testStartFailed, null], [503, C.testStartFailed, null]]) {
    r.reply.start = () => Promise.reject(refusal(status));
    await r.act('run'); await settle();
    assert.deepEqual([r.form().problems, r.form().field, r.model().phase], [[expected], field, 'form'], String(status));
  }
});

test('a refused start\'s reasons give way to a preview asked after them; busy is asked about again 2 s later, a refusal that stays only when the run moves on', async t => {
  // The preview offered the test, the start was refused for a test of this run on its way (pressed in another window):
  // listed with the press's line, the test there asked for (gone meanwhile), the pool asked about again 2 s later, and
  // offered then, nothing is listed any more.
  const e = page(t), p = e.panel;
  await e.open(); await settle(); await e.act('value', '0.02');
  e.reply.start = () => Promise.reject(refusal(409, { detail: 'x', status: 409, reasons: [{ code: 'busy', why: 'test', words: 'x' }] }));
  e.reply.fetch = () => Promise.reject(notFound());
  await e.act('run'); await settle();
  assert.deepEqual([e.form().reasons.map(r => r.code), e.form().problems, e.form().notHere, p.testShown.reasons], [['busy'], [], C.testNotHereFirst, ['busy']]);
  assert.equal(e.previews.length, 1, 'the preview offered it: the press asked nothing more');
  await e.advance(TEST_POOL_AGAIN_MS - 1); assert.equal(e.previews.length, 1); assert.equal(e.fetched, 1, 'the test there asked for');
  await e.advance(1); assert.deepEqual(e.previews.at(-1), [7, 0, true]);
  assert.deepEqual([e.form().reasons, e.form().problems, e.form().notHere, p.testShown.reasons], [[], [], '', []], 'it ended: nothing stands in the way');
  await e.advance(10000); assert.equal(e.previews.length, 2, 'asked about once: nothing loops');
  // Refused without reasons (the same, in the engine's 409 words): "busy", the test there asked for (none), and the pool
  // asked about again 2 s later: the preview's own word then (still on its way), the press's line still said.
  e.reply.start = () => Promise.reject(refusal(409, { detail: 'A test of this run is on its way: wait for its answer, or cancel it', status: 409 }));
  await e.act('run'); await settle();
  assert.deepEqual(e.form().reasons.map(r => r.code), ['busy']);
  e.reply.pool = () => REFUSED([{ code: 'busy', why: 'test', words: 'x' }]);
  await e.advance(TEST_POOL_AGAIN_MS);
  assert.deepEqual([e.fetched, e.previews.length], [2, 3]);
  assert.deepEqual([e.form().reasons.map(r => r.code), e.form().problems, e.form().notHere], [['busy'], [], C.testNotHereFirst]);
  await e.advance(10000); assert.equal(e.previews.length, 3, 'the preview\'s own busy is not asked about by itself: Run asks then');
  // That test ended: a press asks about the pool first, then posts.
  e.reply.pool = () => POOL(); e.reply.start = () => RECORD();
  await e.act('run'); await settle();
  assert.equal(e.previews.length, 4); assert.deepEqual(e.started, [[7, 'cell-compactness', 0.02], [7, 'cell-compactness', 0.02], [7, 'cell-compactness', 0.02]]);
  assert.equal(e.box.dataset.state, 'preparing');

  // A refusal that does not pass by itself (nothing glows in the cell) stays listed, never asked about by itself; the
  // run's end moved (sampled on, paused at another end), the pool is asked for again and its answer is the form's.
  const r = page(t), q = r.panel;
  await r.open(); await settle(); await r.act('value', '0.02');
  r.reply.start = () => Promise.reject(refusal(409, { detail: 'x', status: 409, reasons: [{ code: 'no_suspects', words: 'x' }] }));
  await r.act('run'); await settle(); await r.advance(10000);
  assert.deepEqual([r.previews.length, r.form().reasons.map(x => x.code), r.form().problems, r.form().notHere], [1, ['no_suspects'], [], C.testNotHereCant]);
  r.run({ schedule_done: { iter: 900, seq: 901 } }); await settle(); await settle();
  assert.deepEqual([r.previews.length, r.form().reasons, r.form().problems, r.form().notHere, q.testShown.reasons], [2, [], [], '', []]);
  // ... and a press after a refusal asks about the pool first: refused there, nothing is posted.
  r.reply.start = () => Promise.reject(refusal(409, { detail: 'x', status: 409, reasons: [{ code: 'no_suspects', words: 'x' }] }));
  await r.act('run'); await settle();
  r.reply.pool = () => REFUSED([{ code: 'not_judged', words: 'x' }]);
  await r.act('run'); await settle();
  assert.deepEqual([r.previews.length, r.started.length], [3, 2], 'asked about first, never posted while refused');
  assert.deepEqual([r.form().reasons.map(x => x.text), r.form().problems, r.form().notHere], [['This cell has too few transcripts to be judged'], [], C.testNotHereCant]);

  // A preview asked before the refusal came (on its way while the start was), its answer later, is not the newer word:
  // the refusal stays, and its reason passing by itself, the pool is asked about once that answer is in.
  const o = page(t);
  await o.open(); await settle(); await o.act('value', '0.02');
  let refuse, answer;
  o.reply.start = () => new Promise((_, reject) => { refuse = reject; });
  o.reply.fetch = () => Promise.reject(notFound());
  const pressed = o.act('run'); await settle(); await settle();
  o.reply.pool = () => new Promise(resolve => { answer = resolve; });
  o.run({ schedule_done: { iter: 900, seq: 901 } }); await settle();
  assert.equal(o.previews.length, 2, 'the run\'s end moved: asked for again while the start is on its way');
  refuse(refusal(409, { detail: 'x', status: 409, reasons: [{ code: 'busy', why: 'test', words: 'x' }] })); await pressed; await settle();
  answer(POOL()); await settle(); await settle();
  assert.deepEqual([o.form().reasons.map(x => x.code), o.form().notHere], [['busy'], C.testNotHereFirst], 'asked before the refusal: not its answer');
  o.reply.pool = () => POOL();
  await o.advance(TEST_POOL_AGAIN_MS - 1); assert.equal(o.previews.length, 2);
  await o.advance(1); assert.deepEqual([o.previews.length, o.form().reasons, o.form().problems, o.form().notHere], [3, [], [], '']);

  // Closed: nothing left scheduled for any of them.
  for (const x of [e, r, o]) { if (x.panel.testRec == null) await x.act('close'); }
  assert.deepEqual([r.tasks.size, o.tasks.size], [0, 0]);
  assert.deepEqual(banned(r), []);
});

test('a press while waits hold is taken: the test waits (its line the first wait\'s words), is asked for each second, then its runs start; Cancel ends it', async t => {
  // The run still sampling toward its end, the mixing check off: nothing is asked of the user; Run's tooltip says it.
  const e = page(t, { run: 'running', record: { ...PAIRED, schedule_done: null } }), p = e.panel;
  e.reply.pool = () => OFFERED(['not_at_end', 'check_off'], ['whole_sample']);
  e.reply.start = () => WAITING(['not_at_end', 'check_off']);
  await e.open(); await settle(); await e.act('value', '0.02');
  assert.deepEqual([e.form().reasons, e.form().notHere, e.form().waits.map(w => w.code), e.form().caveats.map(c => c.code)], [[], '', ['not_at_end', 'check_off'],
    ['whole_sample']], 'nothing refused: nothing listed');
  const asks = e.previews.length;
  await e.act('run'); await settle();
  assert.deepEqual(e.started, [[7, 'cell-compactness', 0.02]], 'posted at once');
  assert.equal(e.previews.length, asks, 'the waits\' next ask (2 s later) does not hold the press, nor is it asked first');
  assert.deepEqual([e.box.dataset.state, e.model().phase, e.status().textContent, e.status().dataset.phase], ['waiting', 'waiting', C.testStatusWaiting, 'waiting']);
  assert.deepEqual(e.model().waiting, testWaiting(WAITING(['not_at_end', 'check_off'])));
  assert.deepEqual(e.model().waiting, { text: 'Starts once your run reaches its last iteration',
    tooltip: 'Starts once your run reaches its last iteration\nSwitches the mixing check on first', codes: ['not_at_end', 'check_off'] });
  assert.deepEqual([e.model().form, e.model().progress, e.model().result], [null, null, null], 'no views, no progress yet');
  assert.deepEqual([e.model().test.rule, e.model().test.area, e.model().test.square, e.model().test.cancelTooltip], [null, '', null, C.testCancelWaitingTooltip],
    'no area or run of its own yet');
  // Its Cancel says only what holds then: a run the press resumed is paused again, a mixing check it switched on stays on
  // (never that everything is as it was before the test).
  assert.match(C.testCancelWaitingTooltip, /paused again if the test resumed it; the mixing check stays on\.$/);
  assert.doesNotMatch(C.testCancelWaitingTooltip, /as it was/);
  assert.deepEqual([p.testShown.mode, p.testShown.phase, p.testShown.waits], ['waiting', 'waiting', ['not_at_end', 'check_off']]);
  // Asked for each second while it waits: what it waits for changes as the app does it (the check switched on).
  e.reply.fetch = () => WAITING(['not_at_end']);
  await e.advance(TEST_POLL_MS - 1); assert.equal(e.fetched, 0);
  await e.advance(1); assert.equal(e.fetched, 1);
  assert.deepEqual(e.model().waiting, { text: 'Starts once your run reaches its last iteration', tooltip: '', codes: ['not_at_end'] });
  // A push of its state changes nothing it says (the asks each second do); nothing holds any more: the engine works out
  // the cell's pool before its runs start (minutes for a whole sample), and the line says so.
  e.run({ qc_test: { state: 'waiting', progress: PROGRESS({ phase: 'waiting' }) } });
  e.reply.fetch = () => WAITING([]);
  await e.advance(TEST_POLL_MS); assert.deepEqual(e.model().waiting, { text: C.testPoolAsking, tooltip: '', codes: [] });
  // Its runs start: the views and the progress, as for any test.
  e.reply.fetch = () => RECORD({ state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 12 }), caveats: [{ code: 'whole_sample' }] });
  await e.advance(TEST_POLL_MS);
  assert.deepEqual([e.model().phase, e.model().waiting, e.model().progress.text, e.status().dataset.phase], ['running', null, 'Iteration 12 of 450 · settling', 'running']);
  assert.equal(e.model().test.cancelTooltip, 'Stops the test and deletes its 6 runs. Your run goes on as it was before the test.');
  assert.deepEqual(banned(e), []);
  for (const words of [C.testStatusWaiting, C.testCancelWaitingTooltip]) { assert.deepEqual(bannedWords(words), [], words); assert.ok(clashFree(words), words); }

  // Another run sampling: the press is taken and waits for it; Cancel ends it (the page's DELETE: main.js), the corner shuts.
  const c = page(t);
  c.reply.pool = () => OFFERED(['other_run']); c.reply.start = () => WAITING(['other_run']);
  await c.open(); await settle(); await c.act('value', '0.02'); await c.act('run'); await settle();
  assert.deepEqual([c.model().phase, c.model().waiting.text], ['waiting', 'Starts once the other run pauses or ends']);
  const cancelling = c.act('cancel'); assert.equal(c.model().busyText, C.testCancelling);
  await cancelling; await settle();
  assert.deepEqual([c.cancelled.length, c.cancelled[0][0].state, c.box.dataset.state, c.model()], [1, 'waiting', 'entry', { open: false }]);
  // Close while it waits cancels it too (a test on its way stops with Cancel).
  c.panel.applyTest(WAITING(['check_wait'], { id: 't9' }));
  await c.act('close'); await settle();
  assert.deepEqual([c.cancelled.length, c.cancelled.at(-1)[0].id], [2, 't9']);
  await c.advance(5000); assert.equal(c.tasks.size, 0, 'nothing left to ask');
  // Gone elsewhere (404) while it waits: cancelled, said with Close.
  c.panel.applyTest(WAITING(['details'], { id: 't10' })); c.reply.fetch = () => Promise.reject(notFound());
  await c.advance(TEST_POLL_MS); assert.deepEqual([c.box.dataset.state, c.model().message], ['cancelled', C.testCancelled]);
  // Ended on a refusal while it waited (the run ended before its last iteration): stopped, the reason said, Close.
  c.panel.applyTest(WAITING([], { id: 't11', state: 'failed', reasons: [{ code: 'ended', words: 'x' }], message: 'This run ended before its last iteration' }));
  assert.deepEqual([c.model().phase, c.model().message], ['failed', `${C.testFailed} Reason: This run ended before its last iteration`]);
});

// The short pools' sentence (E3) the fixture's rows with short_rates (cells, as_is) give a pool of k 160 under too_few.
const SHORT_SENTENCE = 'Tests with fewer than 160 suspects, like this one, were measured apart, on only 8 test cells of one sample: about 3 in 100 '
  + 'tests of identical runs came out "clearly fewer" (4 in 100 at worst), and a drop of the suspects alone by 25 % was caught about 2 times in 10.';

test('where the test was not measured it runs all the same; its answer says so in tooltips only: the fewer part\'s ⓘ, else the back part\'s, or the too-few line\'s', async t => {
  const caveats = [{ code: 'whole_sample', words: 'x' }, { code: 'too_few', have: 121, need: 160, words: 'x' }];
  const listed = [{ code: 'whole_sample', text: 'Not measured yet for runs of 20,000 cells or more' }, { code: 'too_few', text: 'Its numbers are for tests with 160 suspects or more: 121 here' }];
  const said = listed.map(x => x.text).join('\n');
  // Nothing else of the answer changes: the parts' words on screen are as without them.
  const strip = a => ({ ...a, caveats: [], fewer: a.fewer && { ...a.fewer, tip: a.calibration }, back: a.back && { ...a.back, info: '' },
    toofew: a.toofew && { ...a.toofew, tip: '' } });
  for (const { name, result } of EXAMPLES) {
    const plain = testReadout({ value: result.setting.value, result: { ...result, caveats: [] } });
    const a = testReadout({ value: result.setting.value, result: { ...result, caveats } });
    assert.deepEqual([plain.caveats, a.caveats], [[], listed], name);
    assert.deepEqual(strip(a), plain, name);
    // E3: a row with short pools' own numbers (cells, as_is) adds them after a too_few caveat.
    const short = result.constants?.short_rates ? SHORT_SENTENCE : '';
    if (a.fewer) assert.deepEqual([a.fewer.tip, a.back.info], [[plain.calibration, said, short].filter(Boolean).join('\n'), ''], `${name}: after how often it is wrong`);
    else if (a.back) assert.equal(a.back.info, said, `${name}: the back part's own ⓘ`);
    if (a.toofew) assert.equal(a.toofew.tip, said, `${name}: the too-few line's tooltip`);
    assert.ok(a.fewer || a.back || a.toofew, `${name}: somewhere to say it`);
    // The example's own caveats, as the engine sent them (a pool judged under k; a test judged with the nearest measured
    // row), said the same way: the page's words for each, never on screen.
    const own = testReadout({ value: result.setting.value, result });
    assert.deepEqual(own.caveats, testCaveats(result.caveats), name);
    assert.deepEqual(strip(own), plain, `${name}: its own caveats change nothing on screen`);
  }
  // The engine's examples where the test was not measured: a pool judged under k carries too_few with the numbers
  // judged (after how often the test is wrong: those numbers are for pools of k or more); a whole sample from nuclei
  // only, narrower than 500 µm one way, its run's three caveats (judged with the nearest measured row, which its result
  // names).
  const short = testReadout({ value: 0.02, result: RESULTS.not_clear_short_pool }), unmeasured = RESULTS.not_clear_unmeasured;
  assert.deepEqual(short.caveats, [{ code: 'too_few', text: 'Its numbers are for tests with 160 suspects or more: 121 here' }]);
  assert.equal(short.fewer.tip, [short.calibration, 'Its numbers are for tests with 160 suspects or more: 121 here'].join('\n'));
  assert.deepEqual([unmeasured.start, unmeasured.situation, unmeasured.constants.start, unmeasured.constants.situation], ['nuclei', 'crop', 'paired', 'square']);
  assert.deepEqual(testReadout({ value: 0.02, result: unmeasured }).caveats.map(x => x.text), ['Not measured yet for runs of 20,000 cells or more',
    'Not measured yet for runs started from nuclei only', 'Not measured yet for regions over 500 µm long and under 500 µm wide']);
  // A code the page doesn't know: "Not measured yet here".
  assert.deepEqual(testReadout({ value: 0.02, result: { ...RESULTS.comes_and_goes, caveats: [{ code: 'a_new_caveat' }] } }).back.info, C.testCaveatUnknown);
  // In the panel's model and the hook: the answer's own caveats, the test's before it.
  const e = page(t), p = e.panel;
  p.applyTest(DONE('comes_and_goes', { result: { ...RESULTS.comes_and_goes, caveats } }));
  assert.deepEqual([e.model().result.caveats, e.model().result.back.info, p.testShown.caveats], [listed, said, ['whole_sample', 'too_few']]);
  assert.deepEqual(banned(e), []);
  for (const x of listed) assert.ok(clashFree(x.text), x.text);
});

test('every reason by its kind: a caveat never stops a press (Run\'s tooltip and the answer\'s say it), a wait is waited for and then the test starts', async t => {
  // Each situation the test was not measured for: posted, never refused; the answer's ⓘ says it, after how often the
  // test is wrong; the hook has its code.
  for (const e of OF_KIND('caveat')) {
    const n = page(t);
    n.reply.pool = () => OFFERED([], [GIVEN(e)]); n.reply.start = () => RECORD({ caveats: [GIVEN(e)] });
    await n.open(); await settle(); await n.act('value', '0.02');
    assert.deepEqual([n.form().reasons, n.form().waits, n.form().caveats], [[], [], [{ code: e.code, text: e.shown }]], e.code);
    await n.act('run'); await settle();
    assert.deepEqual([n.started, n.model().phase, n.panel.testShown.caveats], [[[7, 'cell-compactness', 0.02]], 'preparing', [e.code]], `${e.code}: posted`);
    n.panel.applyTest(DONE('not_clear_apart', { result: { ...RESULTS.not_clear_apart, caveats: [GIVEN(e)] } }));
    assert.equal(n.model().result.fewer.tip, `${calibrationText(RESULTS.not_clear_apart)}\n${e.shown}`, `${e.code}: the answer's ⓘ`);
    assert.deepEqual(n.panel.testShown.caveats, [e.code]);
  }
  // Each prerequisite the app meets itself: posted, the test waits saying for what, then its runs start.
  for (const w of OF_KIND('wait')) {
    const n = page(t);
    n.reply.pool = () => OFFERED([GIVEN(w)]); n.reply.start = () => WAITING([GIVEN(w)]);
    await n.open(); await settle(); await n.act('value', '0.02');
    assert.deepEqual([n.form().reasons, n.form().waits], [[], [{ code: w.code, text: w.shown }]], w.code);
    await n.act('run'); await settle();
    assert.deepEqual([n.started.length, n.model().phase, n.model().waiting.text, n.model().waiting.codes], [1, 'waiting', w.shown, [w.code]], w.code);
    n.reply.fetch = () => RECORD({ state: 'preparing' });
    await n.advance(TEST_POLL_MS);
    assert.deepEqual([n.model().phase, n.model().waiting], ['preparing', null], `${w.code}: then it starts`);
  }
});

test('while the runs work: their progress (the slowest of each side, the schedule\'s parts, what the engine does), the runs and Cancel naming them', async t => {
  assert.deepEqual(testProgress(RECORD()), { iter: 0, of: 450, runs: 6, done: 0, sides: { A: 0, B: 0 }, marks: [200, 400], phase: 'preparing', part: 'settling',
    text: C.testPreparing, tooltip: "Every run starts from your run's starting cells and goes to the end of the schedule: settling to iteration 200, refining to 400, then final settling to 450. The bars show the slowest run as now and the slowest with the value.",
    retry: null });
  const at = (progress, state = 'running') => testProgress(RECORD({ state, progress: PROGRESS(progress) }));
  assert.equal(at({ phase: 'loading' }).text, C.testLoading);
  assert.equal(at({ phase: 'sampling', iter: 42, sides: { A: 42, B: 45 } }).text, 'Iteration 42 of 450 · settling');
  assert.equal(at({ phase: 'sampling', iter: 212, sides: { A: 212, B: 215 } }).text, 'Iteration 212 of 450 · refining');
  assert.equal(at({ phase: 'sampling', iter: 400 }).text, 'Iteration 400 of 450 · final settling');
  assert.deepEqual(at({ phase: 'sampling', iter: 999, sides: { A: 999, B: 451 } }).sides, { A: 450, B: 450 }, 'capped at the end');
  assert.equal(at({ phase: 'writing', iter: 450, done: 4 }).text, 'Writing their results · 4 of 6 runs done');
  assert.equal(at({ phase: 'reading', iter: 450, done: 6 }, 'comparing').text, C.testReading);
  assert.equal(at({ phase: 'sampling' }, 'comparing').text, C.testReading, 'comparing is reading, whatever the phase says');
  assert.equal(at({ retry: true, phase: 'loading' }).retry, C.testRetry);
  assert.equal(testProgress(RECORD({ state: 'running', progress: null })).text, 'Iteration 0 of — · settling', 'never a made-up end');
  // The panel's model while the runs work: the setting under test, its area, the runs and Cancel naming them.
  const e = page(t), p = e.panel;
  p.applyTest(RECORD({ state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 212, sides: { A: 212, B: 215 } }) }));
  const m = e.model();
  assert.deepEqual([m.phase, m.form, m.result], ['running', null, null]);
  assert.deepEqual(m.test, { flag: 'cell-compactness', name: 'Cell compactness', value: 0.02, valueText: '0.02', now: 0.04, nowText: 'now 0.04',
    line: 'Setting: Cell compactness 0.04 → 0.02 (lower = rounder cells)', rule: 'square', area: 'Area: a 500 × 500 µm square around this cell', square: SQUARE.box,
    perSide: 3, runs: 6, runsText: 'Your run waits · 3 runs as now, 3 with 0.02 · about 45 s', how: `${C.testHow} ${C.testHowHold}`,
    cancelTooltip: 'Stops the test and deletes its 6 runs. Your run goes on as it was before the test.' });
  assert.equal(m.progress.text, 'Iteration 212 of 450 · refining'); assert.equal(e.status().textContent, C.testStatusRunning);
  // A start from cells only: 4 + 4 runs.
  p.applyTest(RECORD({ id: 't2', state: 'running', start: 'cells', runs_per_side: 4, area: REGION, square: null, progress: PROGRESS({ runs: 8 }) }));
  assert.deepEqual([e.model().test.runs, e.model().test.rule, e.model().test.square, e.model().test.cancelTooltip],
    [8, 'as_is', null, 'Stops the test and deletes its 8 runs. Your run goes on as it was before the test.']);
  assert.deepEqual(banned(e), []);
});

test('the How tooltip says the runs hold as firmly as your run, then the hold values the engine gave them, as the answer\'s fewer ⓘ does (P4)', async t => {
  // The engine's record (and its answer's result): hold {value (the run's prior-weight at the frame tested), effective,
  // scale, seq, given: {scale, as_now, tried}}; proseg scales the hold by the area's transcripts per cell, so the runs
  // are given the value that holds as firmly there (as now: 1.1615 for the run's 0.5, at scales 4.52 and 10.5).
  const given = { value: 0.5, effective: 5.25, scale: 10.5, seq: 451, given: { scale: 4.52, as_now: 1.1615043878555298, tried: null } };
  const how = `${C.testHow} ${C.testHowHold}`;
  // In a test of another setting every run keeps the run's hold: each is given the value as now.
  assert.equal(holdGiven(given, 'cell-compactness', 0.02), 'Here every run is given 1.1615 for your 0.5.');
  const hold = { ...given, given: { ...given.given, tried: 4.646017551422119 } };
  assert.equal(holdGiven(hold, 'prior-weight', 2), 'Here the runs as now are given 1.1615 for your 0.5. Those with the value are given 4.64602 for 2.');
  assert.equal(holdGiven(hold, 'cell-compactness', 0.02), 'Here every run is given 1.1615 for your 0.5.', 'tried only in a test of the hold');
  assert.equal(holdGiven(given, 'prior-weight', 2), 'Here the runs as now are given 1.1615 for your 0.5.', 'a test of the hold without its tried value');
  for (const none of [null, { value: 0.5, effective: 5.25 }, { ...given, given: null }, { ...given, given: { as_now: null } }]) assert.equal(holdGiven(none, 'prior-weight', 2), '');
  const e = page(t), p = e.panel;
  await e.open();
  assert.equal(e.model().form.how, how, 'the form: before any value is given');
  // While it waits (no run made) and once its runs are given their hold.
  p.applyTest(WAITING(['not_at_end'], { hold: { value: 0.5, effective: 5.25, scale: 10.5, seq: null, given: null } }));
  assert.equal(e.model().test.how, how);
  const words = text => text.split(/\s+/).length;
  assert.ok(words(e.model().test.how) <= 75, `${words(e.model().test.how)} words`);
  p.applyTest(RECORD({ state: 'running', hold: given, progress: PROGRESS({ phase: 'sampling', iter: 12 }) }));
  assert.equal(e.model().test.how, `${how} Here every run is given 1.1615 for your 0.5.`);
  assert.ok(words(e.model().test.how) <= 75, `${words(e.model().test.how)} words`);
  p.applyTest(RECORD({ id: 't3', state: 'running', flag: 'prior-weight', value: 2, current: 0.5, hold, progress: PROGRESS({ phase: 'sampling', iter: 12 }) }));
  assert.equal(e.model().test.how, `${how} Here the runs as now are given 1.1615 for your 0.5. Those with the value are given 4.64602 for 2.`);
  assert.ok(words(e.model().test.how) <= 75, `the How tooltip of a test of the hold, every sentence in: ${words(e.model().test.how)} words`);
  // The answer: its fewer part's ⓘ ends with the same sentences (the result's hold), else as before.
  const r = RESULTS.not_clear_apart, plain = testReadout({ value: r.setting.value, result: r });
  const a = testReadout({ value: r.setting.value, result: { ...r, hold } });
  assert.equal(a.fewer.tip, `${plain.fewer.tip}\nHere every run is given 1.1615 for your 0.5.`);
  const tried = testReadout({ value: 2, result: { ...r, setting: { flag: 'prior-weight', value: 2, current: 0.5 }, hold } });
  assert.ok(tried.fewer.tip.endsWith('\nHere the runs as now are given 1.1615 for your 0.5. Those with the value are given 4.64602 for 2.'));
  assert.deepEqual(banned(e), []);
});

test('the answer, every example of the engine\'s: the parts its verdict has, in the deck\'s words, never derived again from the numbers', () => {
  assert.equal(EXAMPLES.length, 12);
  assert.deepEqual(EXAMPLES.map(e => e.name).sort(), ['clearly_fewer', 'clearly_fewer_shrinking', 'clearly_fewer_suspects_rule', 'comes_and_goes',
    'not_clear_apart', 'not_clear_gap', 'not_clear_negative_gap', 'not_clear_short_pool', 'not_clear_suspects_fire', 'not_clear_unmeasured',
    'too_few_cell_lost', 'too_few_no_controls']);
  for (const { name, result: r } of EXAMPLES) {
    const a = testReadout({ value: r.setting.value, result: r }), v = settingValue(r.setting.value);
    assert.deepEqual([a.verdict, a.rule], [r.verdict, r.rule], name);
    // The parts the verdict has (result v2).
    const has = { too_few: ['toofew'], comes_and_goes: ['back', 'cost', 'went', 'alone'] }[r.verdict] ?? ['back', 'fewer', 'cost', 'went', 'alone'];
    assert.deepEqual(['back', 'fewer', 'cost', 'went', 'alone', 'toofew'].filter(k => a[k]), has, name);
    if (a.back) {
      const numbers = `As now, these ${r.pool.suspects} suspects stay in their cells in ${pc(r.come_back.mean)} % of cases (runs ${r.come_back.held.map(pc).join(' · ')} %).`;
      assert.equal(a.back.title, C.testBackTitle, name);
      if (r.come_back.passed) assert.deepEqual([a.back.word, a.back.answer, a.back.text, a.back.tip], ['yes', C.testBackYes, numbers, ''], name);
      else assert.deepEqual([a.back.word, a.back.answer, a.back.text, a.back.tip], ['no', C.testBackNo, C.testBackNot, numbers], name);
    }
    if (a.fewer) {
      // Only the rule the engine chose is drawn: its title, its verdict word and its reason (fewer.forms stay the record's).
      const excess = r.rule === 'excess', fires = r.verdict === 'clearly_fewer', one = fires && excess && r.cost.shrinking === true;
      assert.equal(a.fewer.title, excess ? C.testFewerTitle : C.testFewerTitleAlone, name);
      assert.equal(a.fewer.word, fires ? 'fewer' : 'unclear', name);
      assert.equal(a.fewer.head, one ? C.testFewerYesShrink : fires ? (excess ? C.testFewerYes : C.testFewerAloneYes) : C.testFewerNo, name);
      assert.equal(a.fewer.why, fires ? 'yes' : r.fewer.why_not, name);
      assert.equal(a.fewer.whyText, { yes: excess ? C.testFewerWhyYes : C.testFewerAloneWhyYes, gap: C.testFewerWhyGap,
        apart: excess ? C.testFewerWhyApart : C.testFewerAloneWhyApart }[a.fewer.why], name);
      assert.equal(a.fewer.shrink, one, name);
      const s = percentPair(r.fewer.suspects.mean.a, r.fewer.suspects.mean.b), o = percentPair(r.fewer.own.mean.a, r.fewer.own.mean.b);
      assert.equal(a.fewer.text, excess ? `With ${v}: suspects ${s[0]} → ${s[1]} %, their own RNA at the same spot ${o[0]} → ${o[1]} %.`
        : `With ${v}: suspects ${s[0]} → ${s[1]} %.`, name);
      // How often the test is wrong, then where it was not measured (the example's own caveats, a line each).
      assert.equal(a.fewer.tip, [a.calibration, ...testCaveats(r.caveats).map(x => x.text)].join('\n'), name);
    }
    if (a.cost) {
      // Three rows (no suspects row): own RNA at the same spot, all their own RNA, transcripts per cell; each change signed.
      assert.deepEqual(a.cost.rows.map(x => [x.key, x.label]), [['own', C.testCostOwn], ['all', C.testCostAll], ['perCell', C.testCostPerCell]], name);
      const signed = c => { const n = Math.round(100 * c); return `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n)} %`; };
      assert.deepEqual(a.cost.rows.map(x => x.change), [r.cost.own_same_spot.change, r.cost.own_all.change, r.cost.transcripts_per_cell.change].map(signed), name);
      assert.equal(a.cost.rows[2].value, `${Math.round(r.cost.transcripts_per_cell.a)} → ${Math.round(r.cost.transcripts_per_cell.b)}`, name);
      // "Mostly by shrinking the cell" whenever the cells shrank, but said once: in the one sentence when it is that.
      const said = r.cost.shrinking === true && !a.fewer?.shrink;
      assert.deepEqual([a.cost.shrink, a.cost.text], [said, said ? C.testShrink : ''], name);
    }
    if (a.went) {
      for (const [k, side] of ['A', 'B'].entries()) {
        const parts = a.went.sides[k].parts;
        assert.equal(a.went.sides[k].side, side, name);
        assert.deepEqual(parts.map(x => [x.key, x.label]), TEST_WENT.map(key => [key, C[{ stay: 'testWentStay', neighbour: 'testWentNeighbour', other: 'testWentOther', background: 'testWentBackground' }[key]]]), name);
        assert.equal(parts.reduce((n, x) => n + x.percent, 0), 100, `${name} ${side}: adds up to 100`);
        for (const x of parts) assert.ok(Math.abs(x.percent - 100 * r.went[side][x.key]) <= 1, `${name} ${side} ${x.key}: within a point of its share`);
      }
    }
    if (a.alone) assert.equal(a.alone.text, r.cell.suspects === 1 ? C.testAloneOne : `This cell alone: ${r.cell.suspects} suspects: too few to judge one cell.`, name);
    if (a.toofew) {
      assert.equal(a.toofew.why, r.why, name);
      assert.equal(a.toofew.text, { pool_short: `Too few suspects were left to judge once the runs were read: ${r.pool.suspects} of ${r.pool.k}.`,
        cell_lost: C.testTooFewCell, no_controls: C.testNoControls }[r.why], name);
    }
    const words = [a.back?.text, a.back?.tip, a.fewer?.head, a.fewer?.text, a.fewer?.whyText, a.calibration, ...(a.cost?.rows ?? []).flatMap(x => [x.label, x.value, x.change]),
      a.cost?.text, ...(a.went?.sides ?? []).map(x => x.text), a.alone?.text, a.toofew?.text].filter(Boolean).join('\n');
    assert.deepEqual(bannedWords(words), [], name); assert.ok(clashFree(words), name);
  }
  // Each kind the spec names is among the examples.
  const r = RESULTS;
  assert.equal(r.comes_and_goes.come_back.passed, false); assert.equal(r.clearly_fewer_suspects_rule.rule, 'suspects');
  assert.equal(r.not_clear_apart.fewer.why_not, 'apart'); assert.equal(r.not_clear_gap.fewer.why_not, 'gap'); assert.ok(r.not_clear_negative_gap.fewer.gap < 0);
  assert.equal(r.clearly_fewer_shrinking.cost.shrinking, true); assert.equal(r.clearly_fewer.cost.shrinking, false);
  assert.equal(r.clearly_fewer_suspects_rule.cell.suspects, 1, 'one suspect: the singular');
  // A pool under what a test needs is judged all the same (no "too few" for it any more), its parts as any other's; the
  // answer's caveat says it (too_few with the numbers judged, engine/qc_test.py answer_caveats).
  assert.ok(r.not_clear_short_pool.pool.suspects < r.not_clear_short_pool.pool.k);
  assert.deepEqual(r.not_clear_short_pool.caveats.map(x => [x.code, x.have, x.need]), [['too_few', r.not_clear_short_pool.pool.suspects, 160]]);
  assert.deepEqual(['back', 'fewer', 'cost', 'went', 'alone', 'toofew'].filter(k => testReadout({ value: 0.02, result: r.not_clear_short_pool })[k]),
    ['back', 'fewer', 'cost', 'went', 'alone']);
  assert.ok(EXAMPLES.every(e => e.result.why !== 'pool_short' && Array.isArray(e.result.caveats)), 'every answer carries its caveats');
  // The two forms disagreeing: the suspects alone fire, the excess (the rule) does not: "Not clear", never "Clearly fewer".
  const fire = testReadout({ value: 0.1, result: r.not_clear_suspects_fire });
  assert.equal(r.not_clear_suspects_fire.fewer.forms.suspects.fires, true); assert.equal(r.not_clear_suspects_fire.fewer.forms.excess.fires, false);
  assert.deepEqual([fire.fewer.head, fire.fewer.word, fire.fewer.title], [C.testFewerNo, 'unclear', C.testFewerTitle]);
  // The verdict is the engine's: numbers that would read otherwise change nothing.
  const twisted = testReadout({ value: 0.02, result: { ...r.not_clear_apart, fewer: { ...r.not_clear_apart.fewer, fires: true, separated: true, gap: 0.5 } } });
  assert.deepEqual([twisted.fewer.head, twisted.fewer.why], [C.testFewerNo, 'apart']);
  // No answer the page reads (another version, none): none said.
  for (const bad of [null, { result: null }, { result: { ...r.clearly_fewer, v: 1 } }, { result: { v: 2 } }]) assert.equal(testReadout(bad), null);
});

test('how often the test is wrong (the answer\'s tooltip): every number from the example\'s own measured row, the settings line only for other settings', () => {
  const tip = r => {
    const p = x => x > 0 ? Math.max(1, Math.round(100 * x)) : 0, from = pc(r.come_back.mean), rates = r.constants.rates;
    return `About ${p(rates.fp_mean)} in 100 tests of identical runs come out "clearly fewer" (${p(rates.fp_worst)} in 100 in the worst of ${rates.areas} areas). `
      + `A drop of the suspects alone by ${Math.round(100 * rates.drop)} % (${from} % → ${Math.round(from * (1 - rates.drop))} %), their own RNA at the same spot unchanged, `
      + `is caught about ${Math.round(10 * rates.power)} times in 10. When the suspects and their own RNA drop alike, it still says "clearly fewer" about `
      + `${p(rates.shrink_fire)} in 100 times. Measured on one sample.`;
  };
  for (const { name, result: r } of EXAMPLES) {
    if (!r.come_back) { assert.equal(calibrationText(r), '', `${name}: nothing to say without the runs as now`); continue; }
    assert.equal(calibrationText(r), r.parent_defaults === false ? `${tip(r)} ${C.testCalibrationSettings}` : tip(r), name);
  }
  // The square's numbers and the region's differ, and each example takes its own.
  const square = RESULTS.not_clear_apart, region = RESULTS.not_clear_negative_gap;
  assert.deepEqual([square.situation, region.situation, square.start, region.start], ['square', 'as_is', 'paired', 'paired']);
  assert.notDeepEqual(square.constants.rates, region.constants.rates);
  assert.notEqual(calibrationText(square).replace(/\(\d+ % → \d+ %\)/, ''), calibrationText(region).replace(/\(\d+ % → \d+ %\)/, ''));
  assert.match(calibrationText(square), /in the worst of 3 areas/); assert.match(calibrationText(region), /in the worst of 2 areas/);
  // "To" is the "from" shown less the drop, so the sentence's own sum holds: 60.5 % reads 61 %, and 61 % less 25 % is 46 %
  // (45.75), not the 45 % that 60.5 % less 25 % would round to.
  assert.equal(RESULTS.clearly_fewer_shrinking.come_back.mean, 0.605);
  assert.match(calibrationText(RESULTS.clearly_fewer_shrinking), /by 25 % \(61 % → 46 %\)/);
  // "At least 1 in 100" for a rate above 0; 0 only for none.
  const tiny = { ...square, constants: { ...square.constants, rates: { ...square.constants.rates, fp_mean: 0.001, shrink_fire: 0 } } };
  assert.match(calibrationText(tiny), /^About 1 in 100 tests/); assert.match(calibrationText(tiny), /about 0 in 100 times/);
  // The settings line only where the run's settings are not proseg's defaults.
  assert.ok(calibrationText(RESULTS.clearly_fewer).endsWith(C.testCalibrationSettings)); assert.equal(RESULTS.clearly_fewer.parent_defaults, false);
  assert.ok(!calibrationText(square).includes(C.testCalibrationSettings));
  // A row the page cannot read: nothing said, never a made-up number.
  for (const rates of [null, { ...square.constants.rates, areas: null }, { ...square.constants.rates, power: 'x' }])
    assert.equal(calibrationText({ ...square, constants: { ...square.constants, rates } }), '');
  for (const text of [calibrationText(square), calibrationText(RESULTS.clearly_fewer)]) { assert.deepEqual(bannedWords(text), [], text); assert.ok(clashFree(text), text); }
});

test("a test with fewer suspects than its numbers are for: the short pools' own numbers follow the too_few caveat in the fewer part's tooltip (E3), only where its row has them", () => {
  // The engine's row for a region as it is from cells only measures pools short of k apart (constants.short_rates:
  // engine/qc_rule.py, 8 test cells of one sample); the caveat too_few stays (the main numbers are full pools'), and the
  // short pools' own sentence follows it: true information only.
  const short = { fp_mean: 0.028426395939086288, fp_worst: 0.037914691943127965, power: 0.21962774957698816, drop: 0.25, shrink_fire: 0.042064297800338414,
    tests: 8, areas: 4 };
  const base = RESULTS.not_clear_short_pool, r = { ...base, constants: { ...base.constants, short_rates: short } };
  const sentence = 'Tests with fewer than 160 suspects, like this one, were measured apart, on only 8 test cells of one sample: about 3 in 100 tests of identical runs came out "clearly fewer" (4 in 100 at worst), and a drop of the suspects alone by 25 % was caught about 2 times in 10.';
  assert.equal(calibrationShortText(r), sentence);
  const a = testReadout({ value: 0.02, result: r }), plain = testReadout({ value: 0.02, result: base });
  assert.equal(a.fewer.tip, [plain.calibration, 'Its numbers are for tests with 160 suspects or more: 121 here', sentence].join('\n'), 'after the caveat');
  assert.equal(plain.fewer.tip, [plain.calibration, 'Its numbers are for tests with 160 suspects or more: 121 here'].join('\n'), 'no short_rates: as before');
  // No too_few caveat (a full pool), or short_rates the page cannot read: nothing added.
  assert.equal(calibrationShortText({ ...r, caveats: [] }), '');
  assert.ok(!testReadout({ value: 0.02, result: { ...r, caveats: [] } }).fewer.tip.includes('measured apart'));
  for (const rates of [null, { ...short, tests: null }, { ...short, power: 'x' }, { ...short, drop: 0 }])
    assert.equal(calibrationShortText({ ...r, constants: { ...r.constants, short_rates: rates } }), '', JSON.stringify(rates));
  assert.equal(calibrationShortText({ ...r, pool: { ...r.pool, k: null } }), '', 'no k: no "fewer than" number');
  assert.equal(calibrationShortText({ ...r, constants: { ...r.constants, short_rates: { ...short, fp_mean: 0.001 } } }).match(/about (\d+) in 100 tests/)[1], '1', 'at least 1 for a rate above 0');
  assert.deepEqual(bannedWords(sentence), []); assert.ok(clashFree(sentence), sentence);
});

test("at a run's end counted for the hold, the pool line and each listed reason say in their tooltips that the test reads the averaged count (testAveraged), never at an averaged end", async t => {
  // Decision A: on a saved F (the pool reply's count "sampled", F's) the panel and the pink dots read the count proseg
  // saves there, while the test judges the cell and picks its suspects on the averaged count (as it was measured).
  const averaged = copy('testAveraged', { i: '450' });
  assert.equal(averaged, "The test judges the cell and picks its suspects on proseg's count averaged over its random background draw; the panel and the pink dots at iteration 450 show the count proseg saves there, so the two can differ a little.");
  const e = page(t);
  e.reply.pool = () => POOL({ count: 'sampled' });
  await e.open(); await settle();
  assert.equal(e.form().pool.tooltip, `${copy('testPoolTooltip', { k: '160' })} ${averaged}`);
  const refused = page(t);
  refused.reply.pool = () => REFUSED(['not_judged', 'same_type'], { count: 'sampled' });
  await refused.open(); await settle();
  assert.deepEqual(refused.form().reasons.map(r => [r.code, r.tip]), [['not_judged', averaged], ['same_type', averaged]]);
  assert.deepEqual(banned(refused), []); assert.ok(clashFree(averaged));
  // At an averaged F (or an engine that names no count) the panel reads the averaged count too: nothing said.
  for (const count of ['averaged', undefined]) {
    const plain = page(t);
    plain.reply.pool = () => POOL(count ? { count } : {});
    await plain.open(); await settle();
    assert.equal(plain.form().pool.tooltip, copy('testPoolTooltip', { k: '160' }), String(count));
    const no = page(t);
    no.reply.pool = () => REFUSED(['not_judged'], count ? { count } : {});
    await no.open(); await settle();
    assert.deepEqual(no.form().reasons, [{ code: 'not_judged', text: testReasonText({ code: 'not_judged' }) }], String(count));
  }
});

test('a cell whose pink dots glow only on the count proseg saves is tested on those: its caveat in Run\'s tooltip, and the pool line\'s tooltip says where its suspects come from (F1)', async t => {
  // The engine refuses no_suspects only when neither count has a glowing dot at F; where the averaged count gives none
  // that can be tested and the count proseg saves does, it takes those, with the caveat saved_suspects (the usual case,
  // suspects on the averaged count that can be tested, unchanged).
  const saved = copy('testAveragedSaved', { i: '450' });
  assert.equal(saved, "The test judges the cell on proseg's count averaged over its random background draw, which gives it none that can be tested, so they are taken from the count proseg saves at iteration 450, as the pink dots show.");
  assert.deepEqual(bannedWords(saved), []); assert.ok(clashFree(saved));
  const e = page(t);
  e.reply.pool = () => OFFERED([], ['saved_suspects'], { count: 'sampled' });
  await e.open(); await settle();
  assert.equal(e.form().pool.tooltip, `${copy('testPoolTooltip', { k: '160' })} ${saved}`);
  assert.ok(!e.form().pool.tooltip.includes(copy('testAveraged', { i: '450' })), 'never "picks its suspects on the averaged count" beside them');
  assert.deepEqual(e.form().caveats, [{ code: 'saved_suspects', text: "Suspects taken from proseg's count as saved: its averaged count gives none that can be tested" }]);
  assert.deepEqual(e.form().reasons, [], 'a caveat: the test runs');
  // ... a refusal that came after its suspects were taken there (here every source of its own kind) says the same.
  const refused = page(t);
  refused.reply.pool = () => REFUSED(['same_type'], { count: 'sampled', caveats: [reason('saved_suspects')] });
  await refused.open(); await settle();
  assert.deepEqual(refused.form().reasons.map(r => [r.code, r.tip]), [['same_type', saved]]);
  // The usual case: suspects on the averaged count, its words as before.
  const usual = page(t);
  usual.reply.pool = () => OFFERED([], ['weak_suspects'], { count: 'sampled' });
  await usual.open(); await settle();
  assert.equal(usual.form().pool.tooltip, `${copy('testPoolTooltip', { k: '160' })} ${copy('testAveraged', { i: '450' })}`);
  assert.deepEqual(banned(e), []); assert.deepEqual(banned(refused), []);
});

test('the answer in the corner: its parts, the answers New run with this, Keep my setting and Run again, none of them the main one', async t => {
  const e = page(t), p = e.panel;
  e.reply.start = () => RECORD({ state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 440 }) });
  await e.open(); await settle(); await e.act('value', '0.02'); await e.act('run'); await settle();
  e.reply.fetch = () => DONE('not_clear_apart');
  await e.advance(1000);
  assert.equal(e.box.dataset.state, 'done'); assert.equal(e.status().textContent, C.testStatusDone);
  const r = e.model().result;
  assert.deepEqual([r.verdict, r.rule, r.repeat], ['not_clear', 'excess', null]);
  assert.deepEqual(r, { ...testReadout(DONE('not_clear_apart')), repeat: null, new: { text: C.testNewRun, tooltip: C.testNewRunTooltip },
    keep: { text: C.testKeep, tooltip: C.testKeepTooltip }, again: { text: C.testRunAgain, tooltip: C.testRunAgainTooltip } });
  assert.equal('primary' in r, false, 'nothing is suggested: no main answer');
  assert.deepEqual([C.testNewRun, C.testKeep, C.testRunAgain], ['New run with this', 'Keep my setting', 'Run again']);
  assert.deepEqual(banned(e), []);
  // Nothing is left to run, nothing slides in the panel; the answer is never asked for again.
  const asked = e.fetched; await e.advance(10000); assert.equal(e.fetched, asked); assert.equal(e.tasks.size, 0, 'nothing scheduled: the page is still');
  assert.equal(e.frames, 0, 'the panel draws no motion of its own for the test');
  // Done without an answer the page reads: the answers all the same, no part.
  p.applyTest(RECORD({ id: 't2', state: 'done', result: null }));
  assert.deepEqual([e.model().result.verdict, e.model().result.back, e.model().result.keep.text], [null, null, C.testKeep]);
});

test('Keep, Cancel and Close go through the page (the run resumes only if it was running: main.js), New run with this calls the page and keeps the test meanwhile', async t => {
  const e = page(t), p = e.panel;
  const answer = async (record, action, reply) => {
    p.applyTest(record);
    e.reply.answer = reply ?? (() => null); e.reply.cancel = reply ?? (() => null);
    const pressed = e.act(action);
    if (p.testRec) assert.equal(e.model().busyText, action === 'cancel' || (action === 'close' && TEST_ACTIVE.includes(record.state)) ? C.testCancelling : C.testWorking);
    await pressed; await settle();
  };
  await answer(DONE(), 'keep');
  assert.deepEqual(e.answered.at(-1), ['t1'], 'the test alone: no "use"'); assert.equal(e.box.dataset.state, 'entry', 'ready for another test');
  assert.deepEqual(e.model(), { open: false }, 'the corner shuts');
  // An answered test the engine still reports (its DELETE on its way, a push) never comes back.
  e.reply.fetch = () => DONE();
  e.run({ qc_test: { state: 'done', progress: PROGRESS() } });
  await e.advance(1000); assert.equal(e.fetched, 1); assert.equal(e.box.dataset.state, 'entry');
  await answer(RECORD({ id: 't3', state: 'running' }), 'cancel');
  assert.equal(e.cancelled.at(-1)[0].id, 't3'); assert.equal(e.box.dataset.state, 'entry');
  // Close ends the test as its state allows: Cancel on its way, Keep for an answer.
  await answer(RECORD({ id: 't8', state: 'running' }), 'close'); assert.equal(e.cancelled.at(-1)[0].id, 't8');
  await answer(DONE('clearly_fewer', { id: 't9' }), 'close'); assert.deepEqual(e.answered.at(-1), ['t9']);
  // What the state does not offer is not done: the answers on a test on its way, Cancel on an answer.
  p.applyTest(RECORD({ id: 't10', state: 'running' }));
  for (const action of ['keep', 'new', 'again']) await e.act(action);
  assert.deepEqual([e.answered.length, e.newRuns, e.box.dataset.state], [2, [], 'running']);
  p.applyTest(DONE('clearly_fewer', { id: 't11' })); await e.act('cancel'); assert.equal(e.cancelled.length, 2);
  // New run with this: the page leaves the run for a new one (the engine ends the test then); here the test stays as it
  // is meanwhile, and a stay (the leave prompt answered "stay") leaves it shown.
  await answer(DONE('clearly_fewer', { id: 't12' }), 'new');
  assert.deepEqual(e.newRuns, ['t12']); assert.equal(e.box.dataset.state, 'done'); assert.equal(e.model().phase, 'done');
  // A refusal: the page's own plain words, else the deck's; the test stays.
  await answer(DONE('clearly_fewer', { id: 't4' }), 'keep', () => Promise.reject(new Error('Failed to fetch')));
  assert.deepEqual([e.box.dataset.state, e.model().message], ['done', C.testAnswerFailed]);
  await answer(RECORD({ id: 't5', state: 'running' }), 'cancel', () => Promise.reject(new Error('500')));
  assert.deepEqual([e.box.dataset.state, e.model().message], ['running', C.testCancelFailed]);
  // A test that stopped: what happened, the engine's reason when it is plain, and Close.
  p.applyTest(RECORD({ id: 't6', state: 'failed', message: 'The test could not finish: a test run did not stop' }));
  assert.equal(e.model().message, `${C.testFailed} Reason: The test could not finish: a test run did not stop`); assert.equal(e.status().textContent, C.testStatusStopped);
  p.applyTest(RECORD({ id: 't7', state: 'failed', message: 'the posterior diverged' }));
  assert.equal(e.model().message, C.testFailed, 'a reason with a banned word is left out'); assert.deepEqual(banned(e), []);
  p.applyTest(RECORD({ id: 't7', state: 'cancelled', message: 'Your run is being saved, so the test stopped' }));
  assert.equal(e.model().message, `${C.testCancelled} Reason: Your run is being saved, so the test stopped`);
  await answer(RECORD({ id: 't7', state: 'cancelled', message: 'x' }), 'close');
  assert.deepEqual(e.answered.at(-1), ['t7']); assert.equal(e.box.dataset.state, 'entry'); assert.deepEqual(e.model(), { open: false });
});

test('Run again: the answer put away as Keep does, the same {cell, flag, value} posted again, and the new answer says the earlier one\'s word and why a repeat is no stronger', async t => {
  const e = page(t), p = e.panel;
  e.reply.start = () => RECORD({ state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 1 }) });
  await e.open(); await settle(); await e.act('value', '0.02'); await e.act('run'); await settle();
  p.applyTest(DONE('clearly_fewer', { value: 0.02 })); p.setCell(null);
  const before = e.models.length;
  e.reply.start = () => RECORD({ id: 't2', state: 'preparing' });
  await e.act('again'); await settle(); await settle();
  assert.deepEqual(e.answered, [['t1']], 'put away as Keep does: the run as it was before the test');
  assert.deepEqual(e.started.at(-1), [7, 'cell-compactness', 0.02], 'the same cell, setting and value, nothing else');
  const phases = e.models.slice(before).map(m => m.open && m.phase);
  assert.deepEqual(phases.filter((x, i) => x !== phases[i - 1]), ['done', 'starting', 'preparing'], 'never shut in between');
  // The new answer: the earlier verdict's word, and the note.
  p.applyTest(DONE('not_clear_apart', { id: 't2', value: 0.02 }));
  assert.deepEqual(e.model().result.repeat, { text: 'Earlier test with 0.02: clearly fewer.', note: C.testRepeatNote });
  // ... the suspects-alone rule's word, lower-cased at its first letter only; nothing judged for the others.
  assert.deepEqual([earlierWord(RESULTS.clearly_fewer), earlierWord(RESULTS.clearly_fewer_suspects_rule), earlierWord(RESULTS.not_clear_gap)],
    ['clearly fewer', 'clearly fewer (own RNA not separated)', 'not clear']);
  for (const name of ['comes_and_goes', 'too_few_cell_lost', 'too_few_no_controls']) assert.equal(earlierWord(RESULTS[name]), null, name);
  assert.equal(earlierWord(RESULTS.not_clear_short_pool), 'not clear', 'a pool under what a test needs is judged');
  e.reply.start = () => RECORD({ id: 't3', state: 'preparing' });
  await e.act('again'); await settle(); await settle();
  p.applyTest(DONE('comes_and_goes', { id: 't3', value: 0.02, flag: 'cell-compactness' }));
  assert.deepEqual(e.model().result.repeat, { text: 'Earlier test with 0.02: not clear.', note: C.testRepeatNote });
  e.reply.start = () => RECORD({ id: 't4', state: 'preparing' });
  await e.act('again'); await settle(); await settle();
  p.applyTest(DONE('not_clear_apart', { id: 't4', value: 0.02 }));
  assert.equal(e.model().result.repeat.text, 'Earlier test with 0.02: nothing judged.');
  // A test that was not a repeat says none.
  p.applyTest(DONE('not_clear_apart', { id: 't9' })); assert.equal(e.model().result.repeat, null);
  // A start refused: the form for that cell, saying why, though no cell is selected.
  p.applyTest(DONE('clearly_fewer', { id: 't5', value: 0.02 }));
  e.reply.start = () => Promise.reject(refusal(500));
  await e.act('again'); await settle(); await settle();
  assert.deepEqual([e.model().phase, e.model().cell, e.form().value, e.form().problems], ['form', { label: 7, name: 'ffaaaaaa-7' }, '0.02', [C.testStartFailed]]);
  // ... changed, it is another test (no earlier word).
  await e.act('value', '0.03'); assert.equal(p.testEarlier, null);
  // Again only for an answer.
  await e.act('close'); p.applyTest(RECORD({ id: 't6', state: 'running' })); const asked = e.answered.length;
  await e.act('again'); assert.equal(e.answered.length, asked);
  for (const text of [copy('testRepeat', { v: '0.02', word: 'clearly fewer' }), copy('testRepeatNone', { v: '0.02' }), C.testRepeatNote]) {
    assert.deepEqual(bannedWords(text), [], text); assert.ok(clashFree(text), text);
  }
});

test('Go deeper\'s "Test a setting here…": the form for the window\'s own cell, the panel shown; a test that is open stays', async t => {
  const e = page(t, { cell: 0 }), p = e.panel;
  assert.equal(e.box.hidden, true, 'no cell selected: no entry');
  p.openTestForm({ label: 12, name: 'ffaaaaaa-12' }); await settle();
  assert.deepEqual([e.box.dataset.state, e.model().cell, e.shows], ['form', { label: 12, name: 'ffaaaaaa-12' }, 1]);
  assert.deepEqual(e.previews, [[12, 0, true]]);
  // A cell selected later takes the form over; none selected again: Go deeper's own form stays.
  p.setCell({ label: 3, name: 'ffaaaaaa-3', color: '#000', line: '' }); assert.deepEqual(e.model().cell, { label: 3, name: 'ffaaaaaa-3' });
  p.openTestForm({ label: 12, name: 'ffaaaaaa-12' }); p.setCell(null); assert.deepEqual(e.model().cell, { label: 12, name: 'ffaaaaaa-12' });
  // A test that is open: its corner shows it; the form is not opened over it (the panel shown all the same).
  p.applyTest(RECORD({ state: 'running' })); p.openTestForm({ label: 4, name: 'ffaaaaaa-4' });
  assert.deepEqual([e.model().phase, e.shows], ['running', 3]);
  for (const bad of [null, { label: 0 }, { label: '4' }]) { const n = e.shows; p.openTestForm(bad); assert.equal(e.shows, n); }
});

test('the hook\'s qc.test (the harness\'s): mode, test, cell, setting, progress, the corner, the start, the area, the reasons, waits and caveats, the pool, the verdict and rule, the hold', async t => {
  const e = page(t, { record: { ...PAIRED, options: { 'prior-weight': 1 }, info: { prior_weight: 1.82 } } }), p = e.panel;
  const KEYS = ['mode', 'id', 'cell', 'flag', 'value', 'current', 'iter', 'of', 'runs', 'done', 'phase', 'corner', 'start', 'area', 'reasons', 'waits', 'caveats',
    'pool', 'verdict', 'rule', 'hold'];
  assert.deepEqual(p.testShown, { mode: 'entry', id: null, cell: 'ffaaaaaa-7', flag: null, value: null, current: null, iter: null, of: null, runs: null, done: null,
    phase: null, corner: false, start: 'paired', area: null, reasons: [], waits: [], caveats: [], pool: null, verdict: null, rule: null,
    hold: { value: 1, effective: 1.82 } });
  e.reply.pool = () => OFFERED(['check_scoring'], ['tiled_box', { code: 'too_few', have: 121, need: 160 }], { area: { rule: 'crop', box: [3150, 3200, 3650, 3380] } });
  await e.open(); await settle(); await e.act('value', '0.02');
  assert.deepEqual(Object.keys(p.testShown), KEYS);
  assert.deepEqual(p.testShown, { mode: 'form', id: null, cell: 'ffaaaaaa-7', flag: 'cell-compactness', value: 0.02, current: f32(0.04), iter: null, of: null,
    runs: null, done: null, phase: 'form', corner: true, start: 'paired', area: 'crop', reasons: [], waits: ['check_scoring'], caveats: ['tiled_box', 'too_few'],
    pool: { cells: 23, suspects: 187, need: 160 }, verdict: null, rule: null, hold: { value: 1, effective: 1.82 } });
  p.applyTest(WAITING(['check_scoring']));
  assert.deepEqual(p.testShown, { mode: 'waiting', id: 't1', cell: 'ffaaaaaa-7', flag: 'cell-compactness', value: 0.02, current: 0.04, iter: 0, of: 450,
    runs: 6, done: 0, phase: 'waiting', corner: true, start: 'paired', area: null, reasons: [], waits: ['check_scoring'], caveats: [], pool: null,
    verdict: null, rule: null, hold: { value: 1, effective: 1.82 } });
  p.applyTest(RECORD({ state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 212, sides: { A: 212, B: 215 } }), caveats: [{ code: 'tiled_box', max_um: 1200 }] }));
  assert.deepEqual(p.testShown, { mode: 'running', id: 't1', cell: 'ffaaaaaa-7', flag: 'cell-compactness', value: 0.02, current: 0.04, iter: 212, of: 450,
    runs: 6, done: 0, phase: 'running', corner: true, start: 'paired', area: 'square', reasons: [], waits: [], caveats: ['tiled_box'],
    pool: { cells: 23, suspects: 187, need: 160 }, verdict: null, rule: null, hold: { value: 1, effective: 1.82 } });
  // Done: the answer's caveats (the pool's, and too few with the judged numbers).
  const r = RESULTS.clearly_fewer_suspects_rule;
  p.applyTest(DONE('clearly_fewer_suspects_rule', { result: { ...r, caveats: [{ code: 'tiled_box', max_um: 1200 }, { code: 'too_few', have: 150, need: 160 }] } }));
  assert.deepEqual([p.testShown.mode, p.testShown.verdict, p.testShown.rule, p.testShown.flag, p.testShown.caveats],
    ['done', 'clearly_fewer', 'suspects', 'prior-weight', ['tiled_box', 'too_few']]);
  // A waiting test that ended on a refusal: its codes.
  p.applyTest(WAITING([], { id: 't2', state: 'failed', reasons: [{ code: 'ended', words: 'x' }], message: 'This run ended before its last iteration' }));
  assert.deepEqual([p.testShown.mode, p.testShown.reasons, p.testShown.waits], ['failed', ['ended'], []]);
});

test('the engine has no test (404): one on its way was cancelled elsewhere and says so; an answer shown was answered elsewhere; one the page does not hold is found', async t => {
  const e = page(t), p = e.panel;
  p.applyTest(RECORD({ state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 30 }) }));
  e.reply.fetch = () => Promise.reject(notFound());
  await e.advance(1000);
  assert.equal(e.box.dataset.state, 'cancelled'); assert.equal(e.model().message, C.testCancelled); assert.equal(e.model().phase, 'cancelled');
  await e.advance(5000); assert.equal(e.fetched, 1, 'nothing more to ask');
  // Another error is not "none": the test is asked for again a second later.
  e.reply.fetch = () => Promise.reject(new Error('Failed to fetch'));
  p.applyTest(RECORD({ id: 't2', state: 'running' }));
  await e.advance(0); assert.equal(e.fetched, 2, 'the last ask was long ago: at once'); assert.equal(e.box.dataset.state, 'running');
  await e.advance(999); assert.equal(e.fetched, 2);
  await e.advance(1); assert.equal(e.fetched, 3); assert.equal(e.box.dataset.state, 'running');
  p.applyTest(DONE('clearly_fewer', { id: 't3' })); e.run({ qc_test: { state: 'cancelled', progress: null } });
  e.reply.fetch = () => Promise.reject(notFound());
  await e.advance(1000); assert.equal(e.box.dataset.state, 'entry'); assert.deepEqual(e.model(), { open: false });
  // A test the page does not hold (a reload, another window) is found from the run's qc_test, for its own cell.
  e.reply.fetch = () => RECORD({ state: 'running', cell: 3, cell_name: 'aabbccdd-1', progress: PROGRESS({ phase: 'sampling', iter: 12 }) });
  e.run({ qc_test: { state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 12 }) } });
  await e.advance(1000); assert.equal(e.box.dataset.state, 'running');
  assert.deepEqual(e.model().cell, { label: 4, name: 'aabbccdd-1' }, 'its own cell, whichever is selected');
  p.setCell(null); assert.equal(e.box.hidden, false, 'a test on its way stays without a selected cell');
  p.setCell({ label: 7, name: 'ffaaaaaa-7', color: '#123', line: '' }); assert.equal(e.find('test-open'), undefined);
  await e.act('run'); assert.deepEqual(e.started, [], 'no second test from the corner either');
  // The run's pushes: their progress is taken as it comes, without asking.
  const asked = e.fetched;
  e.run({ qc_test: { state: 'running', progress: PROGRESS({ phase: 'sampling', iter: 300, sides: { A: 300, B: 301 } }) } });
  assert.equal(e.model().progress.text, 'Iteration 300 of 450 · refining'); assert.equal(e.fetched, asked);
});

test('another run, or none: its test goes, the corner shuts and nothing more is asked; disposed, the corner shuts and nothing is left scheduled', async t => {
  const e = page(t), p = e.panel;
  p.applyTest(RECORD({ state: 'running' })); assert.equal(e.model().open, true);
  p.setTestRun({ id: 'r2', state: 'paused', ...PAIRED });
  assert.equal(e.box.dataset.state, 'entry'); assert.equal(p.testShown.id, null); assert.deepEqual(e.model(), { open: false });
  await e.advance(5000); assert.equal(e.fetched, 0);
  p.setTestRun(null); assert.equal(e.box.hidden, true);
  p.setTestRun({ id: 'r1', state: 'paused', ...PAIRED }); p.applyTest(RECORD({ state: 'running' }));
  assert.ok(e.tasks.size > 0); const count = e.models.length;
  p.dispose(); assert.equal(e.tasks.size, 0);
  assert.deepEqual(e.models.slice(count), [{ open: false }], 'the corner is shut once');
  await e.act('run'); assert.equal(e.models.length, count + 1, 'nothing after');
  // A form whose pool is on its way, disposed: nothing left scheduled, nothing asked after.
  const f = page(t);
  f.reply.pool = () => new Promise(() => {});
  await f.open(); f.panel.dispose(); assert.equal(f.tasks.size, 0);
});

test('nothing in the test section animates by itself', () => {
  const css = readFileSync(new URL('../src/qc.css', import.meta.url), 'utf8');
  assert.doesNotMatch(css, /@keyframes|animation\s*:|transition\s*:|backdrop-filter/);
  assert.match(css, /\.qc-test\b/, 'the section has its styles'); assert.match(css, /\.qc-test-status\b/);
});

// ---- main.js: the test's calls (a slice of the page controller, as the other main.js tests take theirs) -------------
const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const between = (a, b) => { const i = main.indexOf(a), j = main.indexOf(b, i); assert.ok(i >= 0 && j > i, `${a} … ${b}`); return main.slice(i, j); };
/** The page controller's test calls against a scripted engine: `api.deletes` the DELETEs' errors in order (none: it goes
 *  through), `api.replies` their replies in order (the test as it ended), else `ended`'s. */
function controller({ runState = 'paused', knobsValue = f32(0.04), ended = RECORD({ state: 'cancelled' }) } = {}) {
  const calls = [], notices = [], renders = [];
  const context = vm.createContext({
    state: { mode: 'run', run: { id: 'r1', state: runState } }, QC_TEXT: C, copy, settingValue, harness: { counts: {}, count(name) { this.counts[name] = (this.counts[name] ?? 0) + 1; } },
    notice: text => notices.push(text), renderRun: () => renders.push('renderRun'),
    qcPanel: { setTestRun: run => renders.push(['setTestRun', run?.state]), openTestForm: cell => calls.push(['form', cell]) },
    cellName: label => `c${label}`, options: { schema: [{ flag: 'cell-compactness', label: SETTINGS['cell-compactness'].label, min: 0.001, max: 10 }] },
    knobs: { value: () => knobsValue },
    newRunWithSetting: suggestion => { calls.push(['newRun', suggestion]); return Promise.resolve(); },
    api: { deletes: [], replies: [],
      async qcTestPool(id, cell, options) { calls.push(['POOL', id, cell, options.wait, options.signal ?? null]); return POOL(); },
      async qcTestStart(id, body) { calls.push(['POST', id, body]); return RECORD({ id: 'tA' }); },
      async qcTest(id) { calls.push(['GET', id]); return RECORD(); },
      async qcTestCancel(id) { calls.push(['DELETE', id]); const next = this.deletes.shift(); if (next) throw next; return this.replies.shift() ?? ended; },
      async control(id, action) { calls.push(['control', id, action]); if (this.failResume) throw new Error('409'); return { id, state: 'running' }; } },
  });
  vm.runInContext(between('// ---- "Test a setting here…" in the mixing check and in Go deeper, for a cell', '/** The test\'s square on the stage'), context);
  return { context, calls, notices, renders, run: code => vm.runInContext(code, context) };
}

test('main.js: the pool and the start ask for the cell as label - 1; the start carries {cell, flag, value} only; no run means no ask, never "none"', async () => {
  const c = controller({ runState: 'paused' });
  assert.deepEqual(JSON.parse(JSON.stringify(c.run("qcTestSetting('cell-compactness')"))), { label: SETTINGS['cell-compactness'].label, value: f32(0.04), min: 0.001, max: 10 });
  assert.equal(c.run("qcTestSetting('dispersion')"), null);
  await c.run('qcTestPreview(7, { wait: 5, signal: null })');
  await c.run('qcTestPreview(8)');
  const started = await c.run("qcTestStart(7, 'diffusion-probability', 0.3)");
  assert.equal(started.id, 'tA');
  assert.deepEqual(JSON.parse(JSON.stringify(c.calls)), [['POOL', 'r1', 6, 5, null], ['POOL', 'r1', 7, 0, null],
    ['POST', 'r1', { cell: 6, flag: 'diffusion-probability', value: 0.3 }]]);
  assert.deepEqual(c.context.harness.counts, { qcTestPoolFetches: 2, qcTests: 1 });
  await c.run('qcTestFetch()'); assert.deepEqual(c.calls.at(-1), ['GET', 'r1']); assert.equal(c.context.harness.counts.qcTestFetches, 1);
  // Go deeper's button: the panel's form for its cell, named as the page names it.
  c.run('qcTestOpen(9)'); c.run("qcTestOpen(4, 'ffaaaaaa-4')"); c.run('qcTestOpen(0)');
  assert.deepEqual(JSON.parse(JSON.stringify(c.calls.filter(x => x[0] === 'form'))), [['form', { label: 9, name: 'c9' }], ['form', { label: 4, name: 'ffaaaaaa-4' }]]);
  c.run("state.mode = 'pick'");
  await assert.rejects(c.run('qcTestFetch()'), error => error.status !== 404, 'no run: not a 404 (nothing is gone)');
  await assert.rejects(c.run('qcTestPreview(7)'), error => error.status !== 404);
  await assert.rejects(c.run("qcTestStart(7, 'cell-compactness', 0.02)"));
});

test('main.js: Keep, Cancel and Close: the engine forgets the test, and the run resumes only if it was running as the test\'s runs started (never at its end)', async () => {
  const gone = () => Object.assign(new Error('No test'), { status: 404 }), wasRunning = state => RECORD({ state, was_running: true });
  // Keep, the run waiting at its end (the engine's was_running false): DELETE only, left paused.
  let c = controller({ ended: RECORD({ state: 'done' }) });
  await c.run("qcTestEnd({ id: 't1', state: 'done', flag: 'cell-compactness', value: 0.02, current: 0.04, was_running: false })");
  assert.deepEqual(c.calls, [['DELETE', 'r1']]); assert.equal(c.context.state.run.state, 'paused');
  // Was running (sampling on past its end, the engine paused it): resumed.
  c = controller({ ended: wasRunning('done') });
  await c.run("qcTestEnd({ id: 't1', state: 'done', was_running: true })");
  assert.deepEqual(c.calls, [['DELETE', 'r1'], ['control', 'r1', 'resume']]); assert.deepEqual(c.renders, ['renderRun', ['setTestRun', 'running']]);
  // Without the engine's word (neither its reply nor the record says), what this page saw when it started the test;
  // nothing known: left paused.
  c = controller({ runState: 'running', ended: { id: 'tA', state: 'done' } });
  await c.run("qcTestStart(7, 'cell-compactness', 0.02)"); c.run("state.run = { id: 'r1', state: 'paused' }");
  await c.run("qcTestEnd({ id: 'tA', state: 'done' })");
  assert.deepEqual(c.calls.slice(1), [['DELETE', 'r1'], ['control', 'r1', 'resume']]);
  c = controller({ ended: { id: 'tZ', state: 'done' } }); await c.run("qcTestEnd({ id: 'tZ', state: 'done' })"); assert.deepEqual(c.calls, [['DELETE', 'r1']]);
  // Cancel of a test on its way: DELETE cancels it (the engine keeps it, cancelled), a second dismisses it; a 404 is fine.
  c = controller(); await c.run("qcTestEnd({ id: 't1', state: 'running', was_running: false })");
  assert.deepEqual(c.calls, [['DELETE', 'r1'], ['DELETE', 'r1']]); assert.deepEqual(c.renders, []);
  // ... a test still waiting for what comes first too, the run paused (where its press resumed the run, the engine
  // paused it again: was_running false): nothing resumed.
  c = controller(); await c.run("qcTestEnd({ id: 't1', state: 'waiting', was_running: false })");
  assert.deepEqual(c.calls, [['DELETE', 'r1'], ['DELETE', 'r1']]); assert.deepEqual(c.renders, [], 'nothing resumed');
  c = controller(); c.context.api.deletes.push(undefined, gone());
  await c.run("qcTestEnd({ id: 't1', state: 'preparing', was_running: false })"); assert.deepEqual(c.calls, [['DELETE', 'r1'], ['DELETE', 'r1']]);
  // The engine's newest word (its DELETE replies: the test as it ended) before the record this page last saw: a test
  // seen waiting whose runs started since, pausing the run sampling on past its end, resumes it; so does one whose first
  // reply came before the engine paused it (the second's word), or whose second DELETE found it gone (the first's).
  c = controller({ ended: wasRunning('cancelled') }); await c.run("qcTestEnd({ id: 't1', state: 'waiting', was_running: false })");
  assert.deepEqual(c.calls, [['DELETE', 'r1'], ['DELETE', 'r1'], ['control', 'r1', 'resume']]);
  c = controller({ ended: wasRunning('cancelled') }); c.context.api.replies.push(RECORD({ state: 'cancelled' }));
  await c.run("qcTestEnd({ id: 't1', state: 'waiting', was_running: false })");
  assert.deepEqual(c.calls, [['DELETE', 'r1'], ['DELETE', 'r1'], ['control', 'r1', 'resume']]);
  c = controller({ ended: wasRunning('cancelled') }); c.context.api.deletes.push(undefined, gone());
  await c.run("qcTestEnd({ id: 't1', state: 'preparing', was_running: false })");
  assert.deepEqual(c.calls, [['DELETE', 'r1'], ['DELETE', 'r1'], ['control', 'r1', 'resume']]);
  // ... and the engine saying it was not running (it waited at its end) leaves it paused, whatever the record says.
  c = controller({ ended: RECORD({ state: 'done' }) }); await c.run("qcTestEnd({ id: 't1', state: 'done', was_running: true })");
  assert.deepEqual(c.calls, [['DELETE', 'r1']]);
  // A run resumed meanwhile is not resumed again; a resume that fails says so in the general notice.
  c = controller({ runState: 'running', ended: wasRunning('done') }); await c.run("qcTestEnd({ id: 't1', state: 'done', was_running: true })");
  assert.deepEqual(c.calls, [['DELETE', 'r1']]);
  c = controller({ ended: wasRunning('done') }); c.context.api.failResume = true;
  await c.run("qcTestEnd({ id: 't1', state: 'done', was_running: true })"); assert.deepEqual(c.notices, [C.testResumeFailed]);
  // The engine refuses DELETE: the deck's words, nothing resumed.
  c = controller(); c.context.api.deletes.push(Object.assign(new Error('boom'), { status: 500 }));
  await assert.rejects(c.run("qcTestEnd({ id: 't1', state: 'done', was_running: true })"), error => error.plain === C.testAnswerFailed);
  assert.deepEqual(c.calls, [['DELETE', 'r1']]);
  c = controller(); c.context.api.deletes.push(Object.assign(new Error('boom'), { status: 500 }));
  await assert.rejects(c.run("qcTestEnd({ id: 't1', state: 'running', was_running: true })"), error => error.plain === C.testCancelFailed);
  // No Use: nothing is set in the run, whatever is passed.
  c = controller(); await c.run("qcTestEnd({ id: 't1', state: 'done', was_running: false }, true)"); assert.deepEqual(c.calls, [['DELETE', 'r1']]);
});

test('main.js: New run with this fills the picker as a suggestion\'s New run does, with the tested setting at its value from the start', async () => {
  const c = controller();
  await c.run("qcTestNewRun({ id: 't1', state: 'done', flag: 'prior-weight', value: 2, current: 1 })");
  await c.run("qcTestNewRun({ id: 't2', state: 'done', flag: 'diffusion-probability', value: 0.3, current: 0.2 })");
  await c.run("qcTestNewRun({ id: 't3', state: 'done', flag: 'cell-compactness', value: null })");
  assert.deepEqual(JSON.parse(JSON.stringify(c.calls)), [['newRun', { flag: 'prior-weight', to: 2, from: 1 }],
    ['newRun', { flag: 'diffusion-probability', to: 0.3, from: 0.2 }]], 'nothing without a value');
  assert.match(between('function newRunWithSetting(suggestion)', '\n}'), /leave\(/, 'leaving the run: the engine ends its test then');
});

test('main.js wiring: the panel gets the test\'s calls, Go deeper its button and what it reads, the hook the test as shown', () => {
  const panel = between('const qcPanel = new QcPanel(', '});');
  for (const wire of ['testSetting: flag => qcTestSetting(flag)', 'onTestPreview: (label, options) => qcTestPreview(label, options)',
    'onTestStart: (label, flag, value) => qcTestStart(label, flag, value)', 'onTestFetch: () => qcTestFetch()', 'onTestCancel: test => qcTestEnd(test)',
    'onTestAnswer: test => qcTestEnd(test)', 'onTestNewRun: test => qcTestNewRun(test)', 'onTestShow: () => openQcPanel()'])
    assert.ok(panel.includes(wire), wire);
  assert.match(panel, /onTestCorner: model => \{ testCorner\.show\(model\); qcTestSquare\(model\); \}/);
  assert.doesNotMatch(panel, /suspects:|deeper\.suspects/, 'the engine fixes the suspects: the page sends none');
  assert.ok(between('const deeper = new QcDeeper(', '});').includes('onTest: (label, name) => qcTestOpen(label, name)'));
  const context = between('function deeperContext() {', '\n}');
  assert.match(context, /testable: !!run && qcPanel\.testable\?\.\(deeperTestCell\(\)\) === true/);
  assert.match(context, /startHold: \(points, label\) => \(run \? priors\.startHold\?\.\(points, label, run\.info\?\.z \?\? null\) : null\) \?\? null/);
  assert.ok(main.includes('onAction: (action, payload) => qcPanel.testAction(action, payload)') && main.includes('fetchView: id => api.qcTestView(id)'));
  assert.match(between('const later = {', '\n};'), /run: \(\) => paint\.mark\('run', \(\) => \{ renderRun\(\); qcPanel\.setTestRun\?\.\(state\.mode === 'run' \? state\.run : null\); \}\)/);
  assert.match(between('function resetQc(run) {', '\n}'), /qcPanel\.setTestRun\?\.\(run \?\? null\)/);
  // A test's own runs (their meta's qc_test {of, role, side, test}) are never listed, nor opened at boot.
  assert.equal((main.match(/\.filter\(run => !run\.qc_test\?\.of\)/g) ?? []).length, 2);
  assert.ok(between('harness?.provide(', 'boot().catch(').includes('...(qcPanel.testShown ? { test: qcPanel.testShown } : {})'));
  assert.doesNotMatch(main, /\bqcTestHere\b|\bqcUseStopped\b|\bdeeperSources\b|changes: changes\.map/, 'Use, two settings and the sent suspects are gone');
});
