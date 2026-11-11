import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { domFixture } from './dom-fixture.mjs';
import { QcChip, chipModel } from '../src/qc-chip.mjs';
import { QC_TEXT as C, bannedWords, copy, settingChecked, knobName, cardLine } from '../src/qc-copy.mjs';

// The Mixing check panel (qc-panel.js) and the stage chip (qc-chip.mjs) against a minimal DOM: every state in plain
// words, the engine's own numbers, the round-3 slot left empty, actions as callbacks, and motion only when something
// changed and the state allows it. No flagged area is shown since 2026-10-22: the engine's
// areas reach the panel only for its suggestion cards and trials, whose words say "where mixing stands out". Import the .js view as an ES module without a package.json, as the other view tests.
let source = await readFile(new URL('../src/qc-panel.js', import.meta.url), 'utf8');
for (const file of ['qc-copy.mjs', 'qc-chart.mjs', 'explain.mjs']) source = source.replace(`'./${file}'`, JSON.stringify(new URL(`../src/${file}`, import.meta.url).href));
const { QcPanel, PANEL_STATES, fromView, withEngine, suggestionModel, trialRowText, trialVerdictText, headlineInteger,
  ENTRANCE, trialRowDirection, trialVerdictReached } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function browser(t, { reduced = false, saved = null, details = true, options = {} } = {}) {
  const { document, Element } = domFixture();
  const env = { document, now: 0, next: 1, tasks: new Map(), requested: 0, painted: 0, observers: [], calls: [], events: [], frames: 0 };
  Element.prototype.remove = function () {
    if (this.parentElement) this.parentElement.childNodes = this.parentElement.childNodes.filter(n => n !== this);
    this.parentElement = null;
  };
  Element.prototype.removeEventListener = function (type, fn) { this.listeners.set(type, (this.listeners.get(type) ?? []).filter(f => f !== fn)); };
  Element.prototype.getBoundingClientRect = () => { env.layoutReads = (env.layoutReads ?? 0) + 1; return { left: 10, top: 20, width: 218, height: 132 }; };
  Element.prototype.scrollIntoView = function (opts) { env.scrolled = opts; };
  const create = document.createElement;
  document.createElement = tag => {
    const el = create(tag); el.ownerDocument = document;
    if (tag === 'canvas') {
      const context = new Proxy({}, { get: (obj, name) => name in obj ? obj[name] : (...args) => {
        env.calls.push([el, name, ...args]); if (name === 'measureText') return { width: String(args[0]).length * 5 };
      } });
      el.width = 300; el.height = 150; el.getContext = () => context;
    }
    return el;
  };
  document.removeEventListener = Element.prototype.removeEventListener;
  document.hidden = false;
  const media = new Element('media'); media.matches = reduced; env.media = media;
  const storage = new Map(saved == null ? [] : [['sopa-companion:qc:open', String(saved)]]);
  env.storage = { getItem: k => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, v) };
  const add = (fn, delay, kind) => { const id = env.next++; env.tasks.set(id, { fn, at: env.now + delay, kind }); return id; };
  env.scheduler = { now: () => env.now, setTimeout: (fn, ms) => add(fn, ms, 'timer'), clearTimeout: id => env.tasks.delete(id),
    requestAnimationFrame: fn => { env.requested++; return add(fn, 1, 'frame'); }, cancelAnimationFrame: id => env.tasks.delete(id) };
  env.advance = ms => {
    const until = env.now + ms; let count = 0;
    for (;;) {
      const next = [...env.tasks].sort((a, b) => a[1].at - b[1].at)[0];
      if (!next || next[1].at > until) break;
      assert.ok(count++ < 100, 'no runaway render loop'); env.tasks.delete(next[0]); env.now = next[1].at;
      if (next[1].kind === 'frame') env.painted++;
      next[1].fn(env.now);
    }
    env.now = until;
  };
  const globals = { document, window: { devicePixelRatio: 2 }, matchMedia: () => media,
    requestAnimationFrame: env.scheduler.requestAnimationFrame, cancelAnimationFrame: env.scheduler.cancelAnimationFrame,
    ResizeObserver: class {
      constructor(fn) { this.fn = fn; env.observers.push(this); }
      observe(node) { this.node = node; } disconnect() { this.disconnected = true; }
    } };
  const original = new Map(Object.keys(globals).map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { value: v, writable: true, configurable: true });
  t.after(() => { for (const [k, d] of original) if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k]; });
  env.host = document.createElement(details ? 'details' : 'aside'); env.host.open = true; document.body.append(env.host);
  env.callbacks = Object.fromEntries(['onToggle', 'onShowCells', 'onTry', 'onKeep', 'onUndo', 'onNewRun', 'onWhy', 'onInspect', 'onRetry', 'onIdea', 'onCalmMotion', 'onReload']
    .map(name => [name, (...args) => env.events.push([name, ...args])]));
  env.panel = new QcPanel(env.host, { ...env.callbacks, onFrame: () => env.frames++, scheduler: env.scheduler, storage: env.storage, ...options });
  t.after(() => env.panel.dispose());
  env.find = action => env.panel.root.all().find(n => n.dataset.action === action && !n.disabled);
  env.visible = (root = env.panel.root) => root.hidden ? '' : (root.text ?? '') + root.childNodes.map(n => env.visible(n)).join(' ');
  env.clearCount = () => env.calls.filter(c => c[1] === 'clearRect').length;
  return env;
}

const outline = (x, y) => [[x, y], [x + 300, y], [x + 300, y + 300], [x, y + 300]];
/** A state as main.js gives it (qc-panel.js fromView): the engine's summary with its own `shown`. */
function state(per100 = 11, iter = 400, extra = {}) {
  const start = { per100: 10, judged: 3300, median_tx: 386 };
  const areas = [{ L: 'A', tag: 'rose', per100: 23, start_per100: 14, judged: 310, growth: .18, tiny: 4, tiny_start: 1, churn: 2.1, depth: 340, depth_start: 386, outline: outline(0, 0) },
    { L: 'B', tag: 'already', per100: 19, start_per100: 18, judged: 220, growth: .03, outline: outline(500, 0) }];
  const summary = { seq: iter / 10, iter, phase: 'sampling', model: 'm1', per100, shown: Math.round(per100), judged: 3188, median_tx: 340,
    batch_per100: 12, start, areas, more_spots: 2, single: false };
  return { state: 'live', enabled: true, runId: 'run-a', model: 'm1', startSource: 'xenium', summary, start, areas, batch: 12,
    series: [20, 100, 200, iter].map((i, n) => ({ seq: n + 1, iter: i, per100: n === 3 ? per100 : 5 + n * 2, judged: 3188 })), seriesKey: iter,
    changes: [{ iter: 300, key: 'standardShapes' }], formingUntil: 100,
    sampleModel: { cells: 15000, created: '2026-10-17T12:00:00Z', seconds: 70 }, sampleName: 'sample-A', ...extra };
}

test('every panel state says what it is doing in plain words, with a real learning progress bar', t => {
  const e = browser(t), p = e.panel;
  const cases = [
    ['loading', {}, null],
    ['off', {}, C.off],
    ['waiting', { reason: 'gpu_busy', shared: true }, C.waiting],
    ['waiting', { reason: 'lock' }, 'Waiting while another sample is being learned on this computer…'],
    ['waiting', { reason: 'queue', requested: true }, 'Waiting for its turn to learn this sample…'],
    ['learning', { progress: { step: 20, of: 50 }, eta_s: 60 }, "Learning this sample's cell types… 20 of 50 · about 1 min left"],
    ['learning', { progress: { step: 3, of: 429 }, eta_s: null }, "Learning this sample's cell types… 3 of 429"],
    ['waiting-start', {}, C.waitingStart],
    ['live', {}, '11 in 100'],
    ['paused', {}, '11 in 100'],
    ['scrubbing', {}, 'Iteration 400 (not the newest).'],
    ['finished', {}, 'Finished at iteration 400: 11 in 100 cells looked mixed'],
    ['unavailable', { reason: 'no_transcripts' }, C.unavailableNoTranscripts],
    ['unavailable', { reason: 'not_counts' }, C.unavailableCounts],
    ['failed', { reason: 'too_few_cells' }, "Couldn't learn this sample: it has too few cells to learn from."],
    ['unavailable', { reason: 'positions' }, C.unavailablePositions],
    ['failed', { reason: 'something new' }, "Couldn't learn this sample: an unexpected error (details in the engine log)."],
  ];
  for (const [status, extra, text] of cases) {
    p.setState({ ...state(), state: status, ...extra });
    assert.equal(p.root.dataset.state, status, status);
    if (text) assert.ok(e.visible().includes(text), `${status}: ${e.visible().slice(0, 300)}`);
    if (status === 'learning' && extra.eta_s) {
      const bar = p.root.all().find(n => n.tagName === 'PROGRESS'); assert.equal(bar.value, 20); assert.equal(bar.max, 50);
      assert.ok(e.visible().includes(C.learningNote));
    }
  }
  p.setState({ ...state(), state: 'waiting', reason: 'gpu_memory', shared: true, requested: true });
  assert.ok(e.visible().includes(C.cardShared)); assert.ok(e.visible().includes(C.requested));
  p.setState({ ...state(), state: 'learning', progress: { step: 5, of: 50 }, eta_s: 100, relearn: true });
  assert.ok(e.visible().includes('Something this check depends on changed since this sample was learned, so it is learning again (about 2 min).'));
  assert.ok(!e.visible().includes(C.learningNote), 'a relearn is not "once per sample"');
  for (const status of PANEL_STATES) { p.setState({ ...state(), state: status, reason: 'oom' }); assert.equal(p.root.dataset.state, status); }
});

test('switched off, the panel says why a sample cannot be learned, or could not be, in the words it uses when on', t => {
  // With the switch off only "ready" and "off" had a line; a sample the check cannot learn said
  // nothing until the switch was turned on.
  const e = browser(t), p = e.panel;
  p.setState({ ...state(), state: 'off', enabled: false, sampleState: 'unavailable', sampleReason: 'positions' });
  assert.ok(e.visible().includes(C.off) && e.visible().includes(C.unavailablePositions), e.visible().slice(0, 400));
  p.setState({ ...state(), state: 'off', enabled: false, sampleState: 'unavailable', sampleReason: 'no_table' });
  assert.ok(e.visible().includes(C.unavailableTable));
  p.setState({ ...state(), state: 'off', enabled: false, sampleState: 'failed', sampleReason: 'too_few_cells' });
  assert.ok(e.visible().includes("Couldn't learn this sample: it has too few cells to learn from."));
  p.setState({ ...state(), state: 'off', enabled: false, sampleState: 'unavailable', sampleReason: 'a reason not known yet' });
  assert.ok(e.visible().includes(C.unavailable), 'an unknown reason: the general words, never the engine\'s own');
  p.setState({ ...state(), state: 'off', enabled: false, sampleState: 'ready', sampleReason: null });
  assert.ok(e.visible().includes(C.offLearned) && !e.visible().includes(C.unavailable));
  assert.equal(fromView({ status: 'off', sampleState: 'failed', sampleReason: 'oom' }).sampleReason, 'oom');
});

test('where the check cannot learn on this computer the panel says why, switched off or on, and the switch is pressable', t => {
  // No NVIDIA card or Mac GPU it can use, a Mac GPU this PyTorch can't use on its macOS, PyTorch without its CUDA
  // build, a driver too old for it, the packages missing (the
  // probe's reasons, the sample's own), or Clusters and the UMAP that cannot run: each in its own words, never
  // naming the method. The switch is never greyed out: a press turns the run's check on, and the panel says the same.
  const e = browser(t), p = e.panel;
  const words = { no_gpu: C.unavailableGpu, apple_gpu: C.unavailableAppleGpu, torch_cpu: C.unavailableTorchCpu,
    driver: C.unavailableDriver, software: C.unavailableSoftware, analysis: C.unavailableAnalysis };
  for (const [reason, text] of Object.entries(words)) {
    p.setState({ ...state(), state: 'off', enabled: false, sampleState: 'unavailable', sampleReason: reason });
    assert.ok(e.visible().includes(C.off) && e.visible().includes(text), `${reason}: ${e.visible().slice(0, 300)}`);
    assert.equal(p.toggle.disabled, false, `${reason}: the switch stays pressable`);
    const before = e.events.length;
    e.find('toggle').click();
    assert.deepEqual(e.events.slice(before), [['onToggle', true]], reason);
    p.setState({ ...state(), state: 'unavailable', enabled: true, reason, sampleState: 'unavailable', sampleReason: reason });
    assert.ok(e.visible().includes(text), `${reason} (on): ${e.visible().slice(0, 300)}`);
    assert.equal(p.toggle.disabled, false);
    assert.deepEqual(bannedWords(e.visible()), [], reason);
  }
});

test("a run without proseg's own count: the check says why, why a new run has it and an old proseg-live how to update it in the line's tooltip", t => {
  // The engine's unavailable reasons no_counts (made before proseg's own count was kept) and old_proseg (its
  // proseg-live can't keep it; GET /health proseg_fix says how this app finds the program).
  const e = browser(t), p = e.panel, line = () => p.status.childNodes[0];
  p.setState({ ...state(), state: 'unavailable', reason: 'no_counts', prosegFix: 'build' });
  assert.equal(line().textContent, C.unavailableNoCounts);
  assert.equal(line().title, C.unavailableNoCountsTooltip, 'an old run: why it has no count to score, and that a new run has one');
  for (const [fix, words, notebook] of [['build', C.fixBuild], ['install', C.fixInstall], ['setting', C.fixSetting], ['setting', C.fixSettingNotebook, true]]) {
    p.setState({ ...state(), state: 'unavailable', reason: 'old_proseg', prosegFix: fix, notebook });
    assert.equal(line().textContent, C.unavailableOldProseg); assert.equal(line().title, copy('unavailableOldProsegTooltip', { fix: words }), fix);
  }
  p.setState({ ...state(), state: 'unavailable', reason: 'old_proseg', prosegFix: null });
  assert.equal(line().textContent, C.unavailableOldProseg); assert.equal(line().title ?? '', '', 'not known (yet): no tooltip rather than a wrong one');
  assert.equal(p.headline.hidden, true); assert.equal(p.chartBox.hidden, true);
  assert.deepEqual(bannedWords(e.visible()), []);
  assert.equal(fromView({ status: 'unavailable', reason: 'old_proseg' }, { prosegFix: 'install' }).prosegFix, 'install', "the page's own word");
});

test("the headline's ⓘ names proseg's count after the basis and, while this kind of run's thresholds are not re-measured on it, says so", t => {
  const e = browser(t), data = state();
  data.summary.constants = { mode: 'whole', id: 'outline-2026-10', measured: false }; e.panel.setState(data);
  assert.deepEqual(e.panel.numberInfo.title.split('\n').slice(3, 6),
    ["Based on proseg's count (averaged): 3,188 cells with 100 or more transcripts · typical cell 340 (386 at the start).", C.basisTooltip, C.notRemeasured]);
  e.panel.setState({ ...data, summary: { ...data.summary, constants: { ...data.summary.constants, measured: true } } });
  assert.ok(!e.panel.numberInfo.title.includes(C.notRemeasured)); assert.ok(e.panel.numberInfo.title.includes(C.basisTooltip));
  assert.ok(!e.visible().includes(C.basisTooltip) && !e.visible().includes(C.notRemeasured), 'in the ⓘ only');
  // The start line at frame 0 is proseg's count at iteration 0: the same words.
  e.panel.setState({ ...data, state: 'paused', atStart: true, summary: null });
  assert.ok(e.panel.numberInfo.title.includes(`\n${C.basisTooltip}`));
});

test('the engine\'s word on the check as a whole: where Clusters and the UMAP cannot run, the panel says so', () => {
  // GET /health qc: {available, state, reason}. Only "analysis" overrides the run's own answer (nothing is ever scored
  // without Clusters and the UMAP); the check's own reasons come with each run's answer, which is newer.
  const off = { status: 'off', enabled: false, sampleState: 'off', sampleReason: null };
  const on = { status: 'live', enabled: true, reason: null, sampleState: 'ready', sampleReason: null };
  const missing = { available: false, state: 'ready', reason: 'analysis' };
  assert.deepEqual(withEngine(off, missing), { ...off, sampleState: 'unavailable', sampleReason: 'analysis' });
  assert.deepEqual(withEngine(on, missing), { ...on, status: 'unavailable', reason: 'analysis', sampleState: 'unavailable',
    sampleReason: 'analysis' });
  for (const qc of [undefined, null, { available: true, state: 'ready', reason: null },
    { available: false, state: 'unavailable', reason: 'no_gpu' }, { available: false, state: 'unavailable', reason: 'software' }]) {
    assert.equal(withEngine(on, qc), on); assert.equal(withEngine(off, qc), off);
  }
  assert.equal(withEngine(null, missing), null);
  assert.equal(fromView(withEngine(on, missing)).state, 'unavailable');
});

test('without a number (too few cells judged) the transcripts the cells kept since the start stay in view', t => {
  // The retention line lives in the headline, which too few judged cells hide: it went with it.
  const e = browser(t), data = state();
  data.summary.judged = 23; data.summary.coverage = { cells: 91, judged: 23 }; data.summary.retention = { ratio: 0.6 };
  data.series = data.series.map(p => ({ ...p, judged: 23 }));
  e.panel.setState(data);
  assert.equal(e.panel.headline.hidden, true, 'no number from 23 judged cells');
  assert.ok(e.visible().includes('Cells hold 60 % of their start RNA'), e.visible().slice(0, 400));
  e.panel.setState({ ...data, atStart: true, summary: null, start: { per100: 10, judged: 23, cells: 91, median_tx: 40 } });
  assert.ok(!e.visible().includes('of their start RNA'), 'the start line against itself says nothing');
});

test("the headline is the engine's own integer (moved only past a boundary by 0.3), its label and ⓘ, one verdict line beside the start's number, the retention line only when it moves the number", t => {
  const e = browser(t);
  e.panel.setState(state());
  assert.equal(e.panel.numberNode.textContent, '11 in 100');
  // What the number is, what it rests on and the saved results' number are in its ⓘ,
  // an icon beside the label that takes the keyboard; its words are its accessible name too.
  // The start is proseg's count at iteration 0, named so (H3), whatever the cells the run started from.
  const info = e.panel.numberInfo, start = `At the start (proseg's count at iteration 0): 10 in 100. ${C.startTooltip}`;
  const live = copy('numberLive', { i: '400' });
  assert.equal(e.panel.headline.childNodes[0].childNodes.at(-1), info, 'beside the label');
  assert.equal(info.tabIndex, 0); assert.equal(info.getAttribute('aria-label'), info.title); assert.equal(info.getAttribute('role'), 'img');
  assert.equal(info.childNodes[0].tagName, 'SVG', 'an inline icon, not a glyph'); assert.equal(info.textContent, '');
  // ... read on proseg's own count, averaged over its random background draw (its ⓘ after the basis says so), first
  // which iteration it is from and that it is smoothed while live (H10, H2).
  assert.equal(info.title, [live, C.numberTooltip, C.meaning, "Based on proseg's count (averaged): 3,188 cells with 100 or more transcripts · typical cell 340 (386 at the start).",
    C.basisTooltip, "Proseg's cells saved in this sample: 12 in 100."].join('\n'));
  assert.ok(!e.visible().includes(C.meaning) && !e.visible().includes('Based on') && !e.visible().includes('saved in this sample'), 'none of it in view');
  assert.equal(e.panel.estimateNote.hidden, true, 'the foot\'s estimate note would say it twice');
  assert.equal(e.panel.subtitle.hidden, true, 'and the subtitle\'s question: the ⓘ answers it');
  // No like-for-like change from the engine, no change claimed (never the difference of 11 and 10): the start's number
  // alone.
  assert.equal(e.panel.changeNode.hidden, true); assert.equal(e.panel.changeNode.textContent, '');
  assert.equal(e.panel.startNode.textContent, 'At the start: 10 in 100'); assert.equal(e.panel.startNode.title, start);
  // 8.7 smoothed, but the engine still shows 8: the page never rounds the number a second, different way.
  const held = state(8.7); held.summary.shown = 8; e.panel.setState(held);
  assert.equal(e.panel.numberNode.textContent, '8 in 100');
  assert.equal(headlineInteger({ shown: 8, per100: 8.7 }), 8); assert.equal(headlineInteger({ per100: 8.7 }), 9);
  // The headline rose from 10 to 11, but like for like fewer cells look mixed: the engine's change, to one decimal, in one
  // line that says it is like for like (H13: never beside the start's own number, which is iteration 0's); the start's
  // number and that the live change is smoothed in its tooltip.
  const fewer = state(); fewer.summary.change = { raw: 1.1, matched: -1.84, threshold: 0.7, model_threshold: 0.6, meaningful: true, direction: 'fewer' };
  e.panel.setState(fewer);
  assert.equal(e.panel.changeNode.textContent, '▼ 1.8 fewer, like for like'); assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${C.changeSmoothed}`);
  assert.equal(e.panel.startNode.hidden, true, 'the verdict line stands for the start');
  const light = state(); light.summary.change = { raw: 0.6, matched: 0.44, threshold: 0.3, model_threshold: 0.6, meaningful: true, direction: 'more' };
  e.panel.setState(light);
  assert.equal(e.panel.changeNode.textContent, '△ 0.4 more, like for like', "a hollow arrow: by this model's reading");
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${C.changeSmoothed}\n${C.changeLightTooltip}`);
  // 12.49 at the start, 12.51 now (once "▲1"): under the threshold it is about the same.
  const tiny = state(12.51); tiny.start = { ...tiny.start, per100: 12.49 }; tiny.summary.change = { raw: 0.02, matched: 0.02, threshold: 0.7, meaningful: false, direction: 'same' };
  e.panel.setState(tiny); assert.equal(e.panel.changeNode.textContent, 'Same as at the start, like for like');
  const other = state(10.2); other.startSource = 'other'; other.tiled = true; other.summary.change = { matched: 0.1, threshold: 0.7, meaningful: false };
  e.panel.setState(other);
  assert.equal(e.panel.changeNode.textContent, 'Same as at the start, like for like');
  assert.ok(e.panel.changeNode.title.startsWith("At the start (proseg's count at iteration 0): 10 in 100."));
  // The start is proseg's count at iteration 0, the arrow like for like: no note compares it with another count (the
  // whole-sample note against Xenium's own counts went with the outline count).
  assert.ok(!/Xenium/.test(e.panel.changeNode.title), e.panel.changeNode.title);
  e.panel.setState({ ...other, startSource: 'xenium' });
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${C.changeSmoothed}`, 'the same words whatever the cells it started from');
  const bare = state(); bare.start = null; bare.summary.start = null; bare.batch = null; bare.summary.batch_per100 = null;
  bare.summary.change = fewer.summary.change; e.panel.setState(bare);
  assert.equal(e.panel.changeNode.hidden, true, 'no start: no change'); assert.equal(e.panel.startNode.hidden, true);
  assert.equal(e.panel.numberInfo.title, [live, C.numberTooltip, C.meaning, "Based on proseg's count (averaged): 3,188 cells with 100 or more transcripts · typical cell 340.",
    C.basisTooltip].join('\n'), 'no saved results: not said');
  assert.equal(e.panel.retentionNode.hidden, true, 'no retention from the engine: none said');
  // How many of all the run's cells it is based on, and the transcripts the cells kept: in view only when that moves the
  // number by itself, short, what it does in its own ⓘ; otherwise in the headline's ⓘ.
  const counted = state(); counted.summary.coverage = { cells: 3439, judged: 3188, empty: 0, under20: 5, under100: 246, judged_share: 3188 / 3439 };
  counted.summary.retention = { ratio: 0.92, tx: 920000, start_tx: 1000000 }; e.panel.setState(counted);
  assert.ok(e.panel.numberInfo.title.includes(`\nBased on proseg's count (averaged): 3,188 of 3,439 cells (those with 100 or more transcripts).\n${C.basisTooltip}\n`
    + `Cells hold 92 % of the transcripts they started with. ${C.retentionBasis}\n`), e.panel.numberInfo.title);
  assert.equal(e.panel.retentionNode.hidden, true); assert.equal(e.panel.roughNode.hidden, true);
  counted.summary.retention = { ratio: 0.72 }; e.panel.setState({ ...counted });
  assert.equal(e.panel.retentionNode.hidden, false); assert.equal(e.panel.retentionNode.textContent, 'Cells hold 72 % of their start RNA');
  assert.equal(e.panel.retentionInfo.title, `${C.retentionDownTooltip} ${C.retentionBasis}`); assert.ok(!e.panel.numberInfo.title.includes('Cells hold'), 'said once');
  counted.summary.retention = { ratio: 1.23 }; e.panel.setState({ ...counted, summary: { ...counted.summary } });
  // The usual case, cells that gained transcripts, explains a raw number above a like-for-like "fewer".
  assert.equal(e.panel.retentionNode.textContent, 'Cells hold 123 % of their start RNA'); assert.equal(e.panel.retentionInfo.title, `${C.retentionUpTooltip} ${C.retentionBasis}`);
  // Under 300 judged cells the number is rough: said in view, short, the whole basis in the ⓘ.
  const few = state(); few.summary.judged = 120; few.summary.coverage = { cells: 400, judged: 120, empty: 0, under20: 80, under100: 200, judged_share: 0.3 };
  e.panel.setState(few);
  assert.equal(e.panel.roughNode.hidden, false); assert.equal(e.panel.roughNode.textContent, 'Rough: based on only 120 of 400 cells');
  assert.ok(e.panel.numberInfo.title.includes("Based on proseg's count (averaged): only 120 of 400 cells (those with 100 or more transcripts), so this number is rough."));
  assert.ok(!e.visible().includes('fewer transcripts than at the start, which'), 'the old median-based notice is gone');
  e.panel.setState({ ...counted, state: 'paused', atStart: true, summary: null, start: { per100: 10, judged: 3200, cells: 3439, median_tx: 386 } });
  assert.ok(e.panel.numberInfo.title.includes("Based on proseg's count (averaged): 3,200 of 3,439 cells (those with 100 or more transcripts)."), "the start line's own cells");
  assert.ok(e.panel.numberInfo.title.startsWith(`${copy('numberOwn', { i: '0' })}\n`), 'the start line: iteration 0\'s own number');
  assert.equal(e.panel.retentionNode.hidden, true, 'the start against itself says nothing'); assert.equal(e.panel.startNode.hidden, true);
});

test('the summary in view is a number, a few-word verdict and labels; the chart\'s title is its ⓘ', t => {
  const e = browser(t), data = state();
  data.summary.change = { matched: -1.84, threshold: 0.7, model_threshold: 0.6, meaningful: true, direction: 'fewer' };
  data.summary.coverage = { cells: 5400, judged: 5100 }; data.summary.retention = { ratio: 1.15 };
  e.panel.setState(data);
  const words = e.visible(e.panel.headline).trim().split(/\s+/);
  assert.equal(e.visible(e.panel.headline).replace(/\s+/g, ' ').trim(), '11 in 100 cells look mixed ▼ 1.8 fewer, like for like Cells hold 115 % of their start RNA');
  assert.ok(words.length <= 22, `${words.length} words`);
  // The chart: no caption in view, its title an ⓘ in its corner, which says what its dashed line is (H3: proseg's count
  // at iteration 0) while it has one.
  const caption = e.panel.chartBox.childNodes[0];
  assert.equal(caption.tagName, 'FIGCAPTION'); assert.equal(caption.textContent, ''); assert.equal(caption.childNodes[0].title, `${C.chartTitle}\n${C.chartStartTooltip}`);
  e.panel.setState({ ...data, start: null, summary: { ...data.summary, start: null } });
  assert.equal(caption.childNodes[0].title, C.chartTitle, 'no start line: no dashed line to say');
  assert.equal(e.panel.canvas.getAttribute('aria-label'), C.chartTitle);
  // Every ⓘ of the panel: an icon with its words as its tooltip and its name (the number's, the retention line's, the
  // chart's; the areas' went with them, 2026-10-22).
  const infos = e.panel.root.all().filter(n => n.className === 'qc-info');
  assert.ok(infos.length >= 3, `${infos.length} ⓘ`);
  for (const info of infos) { assert.equal(info.getAttribute('aria-label'), info.title); assert.equal(info.tabIndex, 0); }
});

test('nothing that is not a result is shown as one: loading, off, waiting, learning, unavailable and failed', t => {
  const e = browser(t); e.panel.setState(state());
  for (const status of ['loading', 'off', 'waiting', 'learning', 'waiting-start', 'unavailable', 'failed']) {
    e.panel.setState({ ...state(), state: status });
    assert.equal(e.panel.headline.hidden, true, status); assert.equal(e.panel.chartBox.hidden, true, status);
    // "Show on the cells" as long as Colour by: Mixing is offered (the legend says the colours come with the first
    // result); switched off the panel is one compact line: it is not there.
    assert.equal(e.panel.showCells.hidden, !['waiting', 'learning', 'waiting-start'].includes(status), status);
    assert.equal(e.panel.showCells.disabled, status === 'off', status);   // (and out of view while off)
  }
  e.panel.setState({ ...state(), summary: null }); assert.ok(e.visible().includes(C.waitingFrame));
  assert.equal(e.panel.headline.hidden, true);
  const none = state(); none.summary.judged = 0; e.panel.setState(none);
  assert.equal(e.panel.headline.hidden, true); assert.ok(e.visible().includes(C.tooFew));
  const shallow = state(); shallow.summary.median_tx = 60; e.panel.setState(shallow); assert.ok(e.visible().includes(C.lowDepth));
  // Transcripts [0, 0, 200] -> one judged cell of three: once "0 in 100" and no warning.
  const repro = state(0); Object.assign(repro.summary, { judged: 1, mixed: 0, shown: 0, median_tx: 200, coverage: { cells: 3, judged: 1, empty: 2, under20: 0, under100: 0 } });
  e.panel.setState(repro);
  assert.equal(e.panel.headline.hidden, true, 'no headline from one judged cell'); assert.ok(e.visible().includes('Too few cells with 100 or more transcripts yet (1 of 3).'));
  e.panel.setState({ ...repro, areas: [] }); assert.doesNotMatch(e.visible(), /No area stands out/, 'nor "no area stands out" from one judged cell');
  e.panel.setState({ ...repro, areas: [], state: 'finished', runState: 'done' });
  assert.ok(e.visible().includes('Finished at iteration 400, with too few cells of 100 or more transcripts for a number.'));
  assert.ok(!e.visible().includes('cells looked mixed') && !e.visible().includes('No area stood out'), 'no "0 in 100 cells looked mixed"');
  e.panel.setState({ ...repro, areas: [], state: 'finished', runState: 'stopped' });
  assert.ok(e.visible().includes('Stopped at iteration 400, with too few cells of 100 or more transcripts for a number.'));
  const few = state(); few.summary.judged = 29; e.panel.setState(few); assert.equal(e.panel.headline.hidden, true, `under ${29 + 1} judged cells`);
  // Most cells too small to judge: from the counts over all the run's cells, whatever the typical judged cell holds.
  const thin = state(); thin.summary.median_tx = 400; thin.summary.coverage = { cells: 8617, judged: 3100, empty: 1, under20: 98, under100: 5418 };
  e.panel.setState(thin); assert.equal(e.panel.headline.hidden, false); assert.ok(e.visible().includes(C.lowDepth));
  const covered = state(); covered.summary.median_tx = 60; covered.summary.coverage = { cells: 3439, judged: 3188 };
  e.panel.setState(covered); assert.ok(!e.visible().includes(C.lowDepth), 'the counts decide, not the typical judged cell');
});

test('switched off (the default) the panel is one compact line with a real switch: what turning it on does and costs; the chip keeps the stage clear', t => {
  const e = browser(t), chipHost = e.document.createElement('div'), chip = new QcChip(chipHost);
  const off = { ...state(), enabled: false, state: 'off', sampleState: 'off' };
  e.panel.setState(off); chip.setState(off);
  assert.ok(e.visible().includes(C.off));
  // The copy: off, how to turn it on, and what that costs on this sample: learned first (a few minutes, once) or, for a
  // learned sample, results within seconds (it once always warned of minutes).
  assert.match(C.off, /^Off\. Turn it on to see/);
  assert.ok(e.visible().includes(C.offNotLearned)); assert.ok(!e.visible().includes(C.offLearned));
  assert.match(C.offNotLearned, /turning it on learns it first \(a few minutes, once; the run carries on meanwhile\)\.$/);
  e.panel.setState({ ...off, sampleState: 'ready' });
  assert.ok(e.visible().includes(C.offLearned)); assert.ok(!e.visible().includes(C.offNotLearned));
  assert.equal(C.offLearned, 'This sample is already learned: results appear within seconds.');
  e.panel.setState({ ...off, sampleState: 'unavailable' });
  assert.ok(!e.visible().includes(C.offLearned) && !e.visible().includes(C.offNotLearned), 'a sample it cannot learn: no promise');
  e.panel.setState(off);
  // The switch is a real one (a role=switch button, aria-checked; qc.css draws its track and knob) that says what it
  // remembers (it once said "for this run"); nothing else speaks of results that do not exist.
  assert.equal(e.panel.toggle.textContent, C.offLabel); assert.equal(e.panel.toggle.getAttribute('aria-checked'), 'false');
  assert.equal(e.panel.toggle.getAttribute('role'), 'switch');
  assert.equal(e.panel.toggle.title, 'Turn the mixing check on or off. New runs start the way you leave it.');
  assert.equal(e.panel.showCells.hidden, true); assert.equal(e.panel.subtitle.hidden, true);
  assert.equal(e.panel.estimateNote.hidden, true); assert.equal(e.panel.how.hidden, true);
  assert.equal(e.find('show-cells'), undefined, 'nothing to press but the switch'); e.panel.showCells.click(); assert.deepEqual(e.events, []);
  assert.deepEqual(bannedWords([C.off, C.offLearned, C.offNotLearned, C.switchTooltip, C.pickerLearnFirst].join(' ')), []);
  assert.equal(e.panel.headline.hidden, true); assert.equal(e.panel.chartBox.hidden, true);
  assert.equal(e.panel.finished.hidden, true);
  assert.equal(chipHost.hidden, true); assert.equal(chip.label.textContent, '', 'the chip has no words for off: it is hidden');
  assert.equal('chipOff' in C, false); assert.equal('chipUnavailable' in C, false);
  // Even a state object that says "showing cells" never presses it while off.
  e.panel.setState({ ...off, showingCells: true }); assert.equal(e.panel.showCells.getAttribute('aria-pressed'), 'false');
  // Switched on: it can be pressed and says what it does, and the panel says what the check is; before the engine has
  // said, or when the check cannot run, the status says it all.
  e.panel.setState({ ...state(), state: 'learning', progress: { step: 1, of: 50 } });
  assert.equal(e.panel.showCells.hidden, false); assert.equal(e.panel.showCells.disabled, false); assert.equal(e.panel.showCells.title, C.showCellsTooltip);
  assert.equal(e.panel.subtitle.hidden, false); assert.equal(e.panel.estimateNote.hidden, false); assert.equal(e.panel.how.hidden, false);
  e.find('show-cells').click(); assert.deepEqual(e.events.at(-1), ['onShowCells', true]);
  e.panel.setState({ state: 'loading' }); assert.equal(e.panel.showCells.hidden, true);
  e.panel.setState({ ...state(), state: 'failed', reason: 'oom' }); assert.equal(e.panel.showCells.hidden, true);
  e.panel.setState({ ...state(), state: 'unavailable', reason: 'no_transcripts' }); assert.equal(e.panel.showCells.hidden, true);
});

// The selected cell's row (setCell, from main.js updateCellRow every frame): its name, "Go deeper ›", which the page binds
// (qc-deeper.js bindButton: its words, state and click), and its mixing line (qc-copy.mjs cardLine).
const LINE = cardLine(76, 76, 'xenium'), SHORT = cardLine(76, 76, 'xenium', { note: false });
const cell = (label = 7, extra = {}) => ({ label, name: `ffaaaaaa-${label}`, color: `rgb(${label},2,3)`, line: LINE, ...extra });
const bindDeeper = (button, label) => { button.textContent = 'Go deeper ›'; button.dataset.action = 'go-deeper'; button.dataset.cell = String(label); };

test('the selected cell\'s row comes first in the body: its name, "Go deeper ›" and its mixing line, with the check on or off', t => {
  const bound = [];
  const e = browser(t, { options: { bindDeeper: (button, label) => { bound.push(label); bindDeeper(button, label); button.onclick = () => e.events.push(['deeper', label]); } } });
  const p = e.panel, row = p.cellRow;
  assert.equal(p.body.childNodes[0], row, 'first in the body: nothing above it changes while it shows');
  p.setState(state());
  assert.equal(row.hidden, true, 'no cell selected: no row'); assert.equal(e.find('go-deeper'), undefined);
  p.setCell(cell());
  assert.equal(row.hidden, false); assert.equal(row.dataset.cell, 'ffaaaaaa-7', 'its name is data: a word check leaves it out');
  assert.equal(p.cellName.textContent, 'ffaaaaaa-7'); assert.equal(p.cellSwatch.style.background, 'rgb(7,2,3)', 'its colour on the stage');
  assert.deepEqual(row.childNodes.map(n => n.tagName), ['HEADER', 'BUTTON', 'P'], 'the name, the button, then the line');
  assert.equal(p.cellLine.textContent, LINE); assert.equal(p.cellLine.hidden, false); assert.equal(p.cellLine.style.visibility, '');
  assert.ok(e.visible().includes(LINE) && e.visible().includes('ffaaaaaa-7'));
  assert.deepEqual(bound, [7]); assert.equal(e.find('go-deeper'), p.deeperButton);
  e.find('go-deeper').click(); assert.deepEqual(e.events.at(-1), ['deeper', 7]);
  // Off, the panel is compact: the cell and its button stay (the window then offers the check's switch), no line.
  p.setState({ ...state(), enabled: false, state: 'off', sampleState: 'ready' }); p.setCell(cell(7, { line: '' }));
  assert.equal(row.hidden, false); assert.equal(p.cellLine.hidden, true);
  assert.ok(e.visible().includes(C.off) && e.visible().includes('ffaaaaaa-7')); assert.ok(!e.visible().includes('Mixing:'));
  assert.equal(e.find('go-deeper'), p.deeperButton); assert.deepEqual(bound, [7, 7]);
  // Every word it can show passes the check's banned-word rule; nothing selected: no row.
  assert.deepEqual(bannedWords([LINE, SHORT, C.cardNote, cardLine(255, 76, 'xenium')].join(' ')), []);
  p.setCell(null); assert.equal(row.hidden, true);
  // A page that binds no button gets none.
  const bare = new QcPanel(e.document.createElement('details'), { scheduler: e.scheduler, storage: null });
  t.after(() => bare.dispose());
  bare.setCell(cell()); assert.equal(bare.cellRow.hidden, false); assert.equal(bare.deeperButton.hidden, true);
});

test('the row is built once and written only where it changed: results and frames leave its nodes, and the keyboard stays on its button across cells', t => {
  const e = browser(t, { options: { bindDeeper } }), p = e.panel, row = p.cellRow, button = p.deeperButton;
  p.setState(state()); p.setCell(cell());
  const nodes = row.childNodes, kept = [...nodes], words = [p.cellName.childNodes, p.cellLine.childNodes], requested = e.requested;
  button.focus(); assert.equal(e.document.activeElement, button);
  // Results (setState) and frames (setCell, several a frame) as a live run brings them, and the panel's other states.
  for (let k = 1; k <= 5; k++) { p.setState(state(11 + k, 400 + 10 * k)); p.setCell(cell()); p.setCell(cell()); }
  for (const s of [{ state: 'loading' }, { ...state(), state: 'learning', progress: { step: 1, of: 50 } }, { ...state(), enabled: false, state: 'off' }]) {
    p.setState(s); p.setCell(cell());
  }
  assert.equal(row.childNodes, nodes, 'never rebuilt'); assert.deepEqual(row.childNodes, kept); assert.equal(p.deeperButton, button);
  assert.equal(p.cellName.childNodes, words[0], 'its name is not written again'); assert.equal(p.cellLine.childNodes, words[1], 'nor its line');
  assert.equal(e.requested, requested, 'no paint of its own');
  // Another cell: its name and colour, on the same button, which keeps the keyboard.
  p.setCell(cell(9));
  assert.equal(p.cellName.textContent, 'ffaaaaaa-9'); assert.equal(p.cellSwatch.style.background, 'rgb(9,2,3)'); assert.equal(button.dataset.cell, '9');
  assert.equal(row.childNodes, nodes); assert.equal(button.parentElement, row); assert.equal(e.document.activeElement, button);
});

test('"Go deeper ›" sits above the cell\'s line, which never moves it; while the values are on the way a line in the row keeps its place, invisible, and none is made up', t => {
  const e = browser(t, { options: { bindDeeper } }), p = e.panel, line = p.cellLine, button = p.deeperButton;
  p.setState(state());
  // A cell just selected while its values are on the way: no line yet, nor an invisible stand-in for one (an empty box).
  p.setCell(cell(7, { line: '', loading: true }));
  assert.equal(line.hidden, true); assert.equal(line.textContent, ''); assert.equal(line.style.visibility, '');
  // Its values land: the line shows, under the button.
  p.setCell(cell(7));
  assert.equal(line.hidden, false); assert.equal(line.style.visibility, ''); assert.equal(line.textContent, LINE);
  // The next result's values on their way: the line it had keeps its place, invisible, so nothing under the row moves up
  // and back; also when another cell is selected meanwhile, until that cell's own values land.
  p.setCell(cell(7, { line: '', loading: true }));
  assert.equal(line.hidden, false); assert.equal(line.style.visibility, 'hidden'); assert.equal(line.textContent, LINE);
  p.setCell(cell(9, { line: '', loading: true })); assert.equal(line.hidden, false); assert.equal(line.style.visibility, 'hidden');
  const unjudged = cardLine(255, 76, 'xenium');
  p.setCell(cell(9, { line: unjudged })); assert.equal(line.textContent, unjudged); assert.equal(line.style.visibility, '');
  // Nothing on its way (the sample waiting or learning, the check off or unable to run): no line at all.
  p.setCell(cell(9, { line: '', loading: false }));
  assert.equal(line.hidden, true); assert.equal(line.textContent, ''); assert.equal(line.style.visibility, '');
  // A row shown again starts from nothing: no line kept from before it went.
  p.setCell(cell(9)); p.setCell(null); p.setCell(cell(9, { line: '', loading: true }));
  assert.equal(line.hidden, true); assert.equal(line.textContent, '');
  // The line's tooltip: what its words leave out (the panel's meaning line says it under the headline); none without a line.
  p.setCell(cell(9, { line: SHORT, tip: C.cardNote })); assert.equal(line.textContent, SHORT); assert.equal(line.title, C.cardNote);
  p.setCell(cell(9, { line: '', tip: C.cardNote })); assert.equal(line.title, '');
  // Throughout, the button stayed in its row, right under the name and above the line.
  assert.equal(p.cellRow.childNodes[1], button); assert.deepEqual(p.cellRow.childNodes.map(n => n.tagName), ['HEADER', 'BUTTON', 'P']);
});

test('the panel says whether its headline shows: the cell\'s line then leaves out what the headline\'s ⓘ says', t => {
  const e = browser(t), p = e.panel;
  assert.equal(p.headlineShown, false, 'nothing said yet');
  p.setState(state()); assert.equal(p.headlineShown, true); assert.ok(p.numberInfo.title.includes(C.meaning));
  for (const s of [{ ...state(), state: 'learning', progress: { step: 1, of: 50 } }, { ...state(), state: 'waiting', reason: 'gpu_memory' },
    { ...state(), enabled: false, state: 'off' }, { ...state(), summary: { ...state().summary, judged: 20 } }]) {
    p.setState(s); assert.equal(p.headlineShown, false, `${s.state}, judged ${s.summary.judged}`); assert.equal(p.headline.hidden, true);
  }
});

test('frame 0 shows the start line itself, and no word about its areas (none is shown since 2026-10-22); a nuclei-only start has no number', t => {
  const e = browser(t);
  e.panel.setState({ ...state(), state: 'paused', atStart: true, summary: null, areas: [state().areas[1]] });
  assert.equal(e.panel.numberNode.textContent, '10 in 100');
  assert.ok(e.visible().includes('At the start: iteration 0'), 'named as what it is: proseg\'s count at iteration 0 (H3)');
  assert.equal(e.panel.notices.textContent, '', 'no "dashed outlines show ..." (nothing dashed is drawn)');
  assert.equal(e.panel.changeNode.hidden, true, 'the start against itself says nothing'); assert.equal(e.panel.startNode.hidden, true);
  assert.ok(e.panel.numberInfo.title.includes("\nBased on proseg's count (averaged): 3,300 cells with 100 or more transcripts · typical cell 386."));
  assert.doesNotMatch(e.visible(), /Already mixed in|B · 19|\bareas?\b/i, "the start line's areas are not listed");
  assert.equal(e.panel.finished.hidden, true);
  e.panel.setState({ ...state(), state: 'paused', atStart: true, startSource: 'nuclei', start: null, summary: null });
  assert.equal(e.panel.headline.hidden, true); assert.ok(e.visible().includes(C.nucleiStart));
  // A start from cells with their nuclei (the engine's start_kind "paired"): proseg's count at iteration 0 holds the
  // nuclei alone, so no start line either, whatever the engine's start says, and no heading naming the start.
  e.panel.setState({ ...state(), state: 'paused', atStart: true, startKind: 'paired', summary: null });
  assert.equal(e.panel.headline.hidden, true); assert.ok(e.visible().includes(C.nucleiCellsStart)); assert.ok(!e.visible().includes(C.nucleiStart));
  assert.ok(!e.visible().includes('At the start'), e.visible());
  e.panel.setState({ ...state(), startKind: 'paired' });
  assert.equal(e.panel.numberNode.textContent, '11 in 100', 'its own number, as any frame');
  assert.equal(e.panel.startNode.hidden, true); assert.equal(e.panel.changeNode.hidden, true); assert.ok(e.visible().includes(C.nucleiCellsStart));
  // A start from cells alone has its start line (start_kind "cells"), as with an engine that names no start_kind.
  e.panel.setState({ ...state(), startKind: 'cells' });
  assert.equal(e.panel.startNode.textContent, 'At the start: 10 in 100'); assert.ok(!e.visible().includes(C.nucleiCellsStart));
  assert.equal(e.requested, 0);
});

test('scrubbing names the iteration of the result shown and selects it on the chart; no word about its areas', t => {
  const e = browser(t); e.panel.setState(state(12));
  const back = state(8.4, 200); back.state = 'scrubbing'; back.summary.single = true; back.areaView = 'single';
  back.areas = [{ L: 'C', tag: 'similar', per100: 18, start_per100: 17, judged: 100, outline: outline(0, 0) }];
  e.panel.setState(back);
  assert.equal(e.panel.numberNode.textContent, '8 in 100'); assert.ok(e.visible().includes('Iteration 200 (not the newest).'));
  assert.equal(e.panel.layout.selected.iter, 200, 'the chart marks the iteration shown');
  assert.doesNotMatch(e.visible(), /C · 18|\bareas?\b|smoothed/i, "neither its areas nor that they are this iteration's alone");
  const live = state(12.2); live.summary.single = true; live.areaView = 'single'; e.panel.setState(live);
  assert.equal(e.panel.notices.textContent, '', 'a single snapshot has nothing to say: no area is shown');
});

test("looking back, a result of another iteration than the frame on screen (the nearest with proseg's count) says which, beside the number", t => {
  const e = browser(t), at = copy('snappedTooltipMixing', { i: '200' });
  // Scrubbed to iteration 237, which has no count of its own: its status names the frame's iteration, the badge the
  // result's, why in its tooltip.
  e.panel.setState(fromView({ status: 'scrubbing', enabled: true, runId: 'r', model: 'm', summary: state(8.4, 200).summary, start: state().start,
    startSource: 'xenium', atStart: false, line: state().series, lineVersion: 1, ticks: [], formingUntil: 100, areas: [], snapped: 200, frameIter: 237 }));
  assert.ok(e.visible().includes('Iteration 237 (not the newest).'), e.visible().slice(0, 200));
  assert.equal(e.panel.snappedNode.hidden, false); assert.equal(e.panel.snappedNode.textContent, 'At iteration 200'); assert.equal(e.panel.snappedNode.title, at);
  assert.equal(e.panel.headline.childNodes[1], e.panel.snappedNode, 'right under the number');
  // Paused with the paused frame's own result still on its way: the badge alone.
  e.panel.setState({ ...state(8.4, 1200), state: 'paused', snapped: 1200, frameIter: 1250 });
  assert.equal(e.panel.snappedNode.textContent, 'At iteration 1,200'); assert.ok(!e.visible().includes('not the newest'));
  // Its own result, following live (the view says none then), frame 0 and no number: no badge.
  for (const other of [{ ...state(), snapped: null }, { ...state() }, { ...state(), state: 'paused', atStart: true, summary: null, snapped: 30 }]) {
    e.panel.setState(other); assert.equal(e.panel.snappedNode.hidden, true); assert.equal(e.panel.snappedNode.textContent, ''); assert.equal(e.panel.snappedNode.title, '');
  }
  const few = state(); few.summary.judged = 12; e.panel.setState({ ...few, snapped: 400 }); assert.equal(e.panel.headline.hidden, true);
  assert.deepEqual(bannedWords(copy('snappedBadge', { i: '1,200' }) + at), []);
  // The chip on the stage says it in its tooltip, after the frame's own iteration.
  const chip = chipModel({ ...state(8.4, 200), state: 'scrubbing', snapped: 200, frameIter: 237 });
  assert.ok(chip.tooltip.startsWith(`Iteration 237 (not the newest). ${at} `), chip.tooltip);
  assert.ok(chipModel({ ...state(8.4, 200), state: 'paused', snapped: 200, frameIter: 237 }).tooltip.startsWith(`${at} `));
  assert.ok(!chipModel({ ...state(), state: 'paused' }).tooltip.includes('nearest with'));
  // Scrubbing on between two frames that show the same result: the status line and the chip follow the frame on screen.
  const chipHost = e.document.createElement('div'), qcChip = new QcChip(chipHost), scrubbed = { ...state(8.4, 200), state: 'scrubbing', snapped: 200 };
  for (const frameIter of [237, 240]) {
    e.panel.setState({ ...scrubbed, frameIter }); qcChip.setState({ ...scrubbed, frameIter });
    assert.ok(e.visible().includes(`Iteration ${frameIter} (not the newest).`), String(frameIter));
    assert.ok(qcChip.root.title.startsWith(`Iteration ${frameIter} (not the newest). ${at} `), qcChip.root.title);
  }
  assert.ok(!e.visible().includes('Iteration 237'));
  // ... and the chip's badge follows the result shown (paused: its tooltip alone names it).
  qcChip.setState({ ...state(8.4, 190), state: 'paused', snapped: 190, frameIter: 240 });
  assert.ok(qcChip.root.title.startsWith(`${copy('snappedTooltipMixing', { i: '190' })} `), qcChip.root.title);
  assert.doesNotMatch(qcChip.root.title, /Clusters/, 'the check names its own iteration, never the clusters\'');
});

test("the slot is in place and empty: no suggestion card, trial readout or previous run until it is switched on", t => {
  const e = browser(t), data = state();
  data.summary.suggestion = { rule: 'R4', areas: ['A'], flag: 'cell-compactness', from: .04, to: .02, timing: 'later' };
  data.trial = { rule: 'R4', flag: 'cell-compactness', from: .04, to: .02, area: 'A', start_iter: 200, rows: [], verdict: null };
  data.previous = { run: 'old', series: [{ iter: 0, per100: 9 }, { iter: 500, per100: 11.5 }] };
  e.panel.setState(data);
  assert.equal(e.panel.slot.hidden, true); assert.equal(e.panel.slot.dataset.slot, 'suggestions');
  assert.equal(e.panel.slot.textContent, ''); assert.equal(e.find('try'), undefined);
  assert.ok(!e.visible().includes(C.suggestions)); assert.ok(!e.visible().includes(C.tooEarly));
  assert.equal(e.panel.layout.band.length, 0); assert.equal(e.panel.layout.previousPoints, undefined);
});

// The suggestion card, with the engine's card as engine/qc_rules.py
// evaluate gives it and the Live knobs labels of engine/proseg_options.py (the hold's says what it does whatever the run
// started from).
const LABELS = { 'cell-compactness': 'Cell compactness (lower = rounder cells)', 'prior-weight': 'Prior weight (higher = the starting cells hold their transcripts harder)',
  'no-diffusion': 'Switch off the transcript drift model (no diffusion)' };
const knobLabel = flag => LABELS[flag] ?? null;
const r4 = (timing = 'early', extra = {}) => ({ id: 'R4:cell-compactness:0.02', rule: 'R4', areas: ['A'], flag: 'cell-compactness', from: 0.04, to: 0.02,
  confidence: 'rough', timing, actions: timing === 'early' ? ['try', 'new_run', 'why'] : ['new_run', 'try', 'why'], main: timing === 'early' ? 'try' : 'new_run',
  growth: 0.18, ...extra });
const cardOf = e => e.panel.suggestions.all().find(n => n.className === 'qc-suggestion-card');
const inOrder = (text, parts) => { let at = -1; for (const part of parts) { const i = text.indexOf(part, at + 1); assert.ok(i > at, `${part} in ${text}`); at = i; } };
const suggesting = (e, data) => e.panel.setState(data);
/** The words of a card in view, its buttons aside (shorter cards, 2026-10-22: about 25). */
const cardWords = card => {
  const words = n => n.hidden || n.tagName === 'BUTTON' ? '' : [n.text ?? '', ...n.childNodes.map(words)].join(' ');
  return words(card).split(/\s+/).filter(w => /\p{L}/u.test(w)).length;
};

test('one card while cells form: Try it first, the Live knobs name, never the raw flag', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state(11, 60);
  data.summary.phase = 'burnin'; data.suggestion = r4('early');
  suggesting(e, data);
  const card = cardOf(e);
  assert.equal(card.dataset.rule, 'R4');
  // 2026-10-22 (less text and shorter cards): in view the title, one short why, the
  // Setting line, the confidence and the buttons; the cost, the timing, the tests behind it and the footer under Why?,
  // in that order.
  const inView = e.visible(card);
  inOrder(inView, [C.ruleR4Title, 'Cells where mixing stands out grew about 18 % bigger and seem to reach into their neighbours.', 'Cell compactness 0.04 → 0.02', 'Rough guess',
    C.try, C.newRun, C.why]);
  for (const text of [C.shapeCost, C.timingEarly, C.ruleR4Why, C.suggestionFooter, 'Setting:', 'lower = rounder cells']) assert.ok(!inView.includes(text), `under Why?: ${text}`);
  assert.ok(cardWords(card) <= 25, `${cardWords(card)} words in view`);
  inOrder(card.textContent, [C.why, C.shapeCost, C.timingEarly, C.ruleR4Why, C.suggestionFooter]);
  // No "for area A" line (2026-10-22): the why says where; no letter anywhere in the card.
  assert.equal(card.all().some(n => n.className === 'qc-small qc-card-area'), false); assert.doesNotMatch(card.textContent, /for areas?|\bin A\b/);
  // The Live knobs label's hint is the Setting line's ⓘ.
  const setting = card.all().find(n => n.className === 'qc-setting'), hint = setting.all().find(n => n.className === 'qc-info');
  assert.equal(hint.title, 'lower = rounder cells'); assert.equal(hint.getAttribute('aria-label'), hint.title); assert.equal(hint.tabIndex, 0);
  assert.ok(!setting.title, 'no tooltip with the raw flag (the Live knobs name finds the knob)');
  assert.ok(!setting.textContent.includes('cell-compactness'));
  const tryButton = e.find('try'), newRun = e.find('new-run');
  assert.equal('primary' in tryButton.dataset, true, 'while cells form Try it is the main action'); assert.ok(!newRun.className.includes('primary'));
  assert.equal(tryButton.title, 'Sets Cell compactness to 0.02 for the whole run from the next iteration. Undo puts back 0.04.');
  assert.equal(newRun.title, C.newRunTooltip);
  tryButton.click(); newRun.click();
  assert.deepEqual(e.events.map(x => x[0]), ['onTry', 'onNewRun']); assert.equal(e.events[0][1], data.suggestion, 'the engine\'s own card');
  assert.ok(!e.visible().includes(C.ruleR4Why), 'Why? is folded'); assert.equal(e.find('why').getAttribute('aria-expanded'), 'false');
  e.find('why').click(); assert.ok(e.visible().includes(C.ruleR4Why)); assert.equal(e.find('why').getAttribute('aria-expanded'), 'true');
  inOrder(e.visible(card), [C.why, C.shapeCost, C.timingEarly, C.ruleR4Why, C.suggestionFooter]);
  e.find('why').click(); assert.ok(!e.visible().includes(C.shapeCost), 'folded again');
  assert.deepEqual(e.panel.shown.card, { rule: 'R4', areas: ['A'], flag: 'cell-compactness', to: 0.02, actions: ['try', 'new-run'], primary: 'try' });
  assert.equal(e.requested, 0, 'the first card of a run appears without motion');
});

test('once cells have formed, New run with this setting is the main action; the timing line names the time here', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  data.suggestion = r4('later', { later_s: 162 });
  suggesting(e, data);
  const card = cardOf(e), buttons = card.all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action);
  assert.deepEqual(buttons, ['new-run', 'try', 'why'], 'New run first, Try it stays offered');
  assert.ok(e.find('new-run').className.includes('primary')); assert.equal('primary' in e.find('new-run').dataset, true);
  assert.ok(card.textContent.includes(copy('timingLater', { t: '3 min' })), 'about 3 min here (the engine\'s later_s)');
  assert.ok(!e.visible(card).includes(copy('timingLater', { t: '3 min' })), 'under Why?');
  // The engine's own main action wins; without one the phase decides; a card without actions has none of these buttons.
  assert.deepEqual(suggestionModel({ ...r4('later'), main: 'try' }, data).actions, ['try', 'new-run']);
  assert.equal(suggestionModel({ ...r4('later'), actions: undefined, main: undefined }, data, { phase: 'sampling' }).primary, 'new-run');
  assert.equal(suggestionModel({ ...r4('later'), actions: undefined, main: undefined, timing: undefined }, data, { phase: 'burnin' }).primary, 'try');
  assert.equal(suggestionModel(r4('later'), data, { laterTime: '9 min' }).timing, copy('timingLater', { t: '9 min' }), 'the page\'s time without the card\'s');
  assert.equal(suggestionModel(r4('later', { later_s: null }), data).timing, C.timingLaterNoTime);
});

test('every rule in its own words; R4 a rough guess; U, R1, R6 and R7 without a knob button; no jargon anywhere', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  // Each card: what is in view (shorter cards, 2026-10-22: its title, one short why, the
  // Setting line, the confidence, the buttons), what folds under Why? (the why's other sentences, the cost, the timing,
  // the tests behind it, the footer's caveats), the Setting line's ⓘ and the buttons.
  const cases = {
    U: [{ id: 'U:cell-compactness:0.04:1', rule: 'U', areas: [], flag: 'cell-compactness', from: 0.02, to: 0.04, changed: { flag: 'cell-compactness', from: 0.04, to: 0.02 },
      before_per100: 11, after_per100: 12.6, confidence: 'seen', timing: 'later', actions: ['undo', 'keep'], main: 'undo' },
      [C.ruleUTitle, 'Like for like, mixing rose from 11.0 to 12.6 in 100 after you changed Cell compactness.', 'Cell compactness 0.02 → 0.04',
        C.seen, C.undoIt, C.keep, C.why], [C.suggestionFooterWhole], 'lower = rounder cells', ['undo-suggestion', 'keep-suggestion', 'why']],
    // The engine's "unlikely" is never shown for a card that offers no setting: it says where to look.
    R1: [{ rule: 'R1', areas: ['B'], confidence: 'unlikely', timing: 'later', actions: ['show_start'], main: 'show_start' },
      [C.ruleR1Title, 'These cells already looked mixed at iteration 0, and proseg has not made them more mixed.', C.closerLook,
        "Show the outlines of Xenium's cells here", C.why],
      ["The cause isn't known", 'check the tissue image and marker genes before naming their cell type', C.suggestionFooterInfo], null, ['show-start', 'why']],
    R2: [{ rule: 'R2', areas: ['A', 'C'], flag: 'cell-compactness', from: 0.08, to: 0.04, v: 0.08, confidence: 'likely', timing: 'later', actions: ['new_run', 'try', 'why'], main: 'new_run' },
      [C.ruleR2Title, 'Shapes are looser than standard (0.08); cells where mixing stands out grew about 18 % bigger.', 'Cell compactness 0.08 → 0.04', C.likelySize,
        C.newRun, C.try, C.why], [C.shapeCost, C.timingLaterNoTime, C.ruleR2Why, C.suggestionFooterNewRun], 'lower = rounder cells', ['new-run', 'try', 'why']],
    // Drift is judged live on proseg's count, and its size was measured there: likely, about 1 in 100, as R2.
    R3: [{ rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'likely', timing: 'later', actions: ['try', 'why'], main: 'try' },
      [C.ruleR3Title, 'With drift off, cells kept 7-17 % fewer transcripts in tests.', 'Untick "Switch off the transcript drift model"', C.likelySize,
        C.try, C.why], [C.ruleR3More, C.ruleR3Why, C.suggestionFooterDrift], 'drift on', ['try', 'why']],
    R5: [{ rule: 'R5', areas: ['A'], flag: 'prior-weight', from: 1, to: 2, k: 4, confidence: 'rough', timing: 'later', actions: ['new_run', 'try', 'why'], main: 'new_run' },
      [C.ruleR5Title, 'Small cells where mixing stands out are squeezed by their neighbours: 4 have shrunk to almost nothing.', 'Prior weight 1 → 2', C.rough, C.newRun, C.try, C.why],
      [C.timingUntested, C.ruleR5Why, C.suggestionFooterNewRun], 'higher = the starting cells hold their transcripts harder', ['new-run', 'try', 'why']],
    R6: [{ rule: 'R6', areas: ['A'], confidence: 'unlikely', timing: 'later', actions: [], main: null },
      [C.ruleR6Title, 'The outlines may not be what changed: RNA may be spreading in the tissue.', C.closerLook, C.why],
      ['Where mixing stands out, cells got more mixed without growing or shrinking.', "The cause isn't known; check the tissue image and marker genes there.", C.suggestionFooterInfo], null, ['why']],
    R7: [{ rule: 'R7', areas: ['A'], tried: { flag: 'prior-weight', from: 1, to: 2 }, confidence: 'unlikely', timing: 'later', actions: [], main: null },
      [C.ruleR7Title, 'You tried a firmer hold to the starting cells here and mixing rose rather than dropped.', C.seen, C.why],
      ['Other settings were not tried here.', "Its cells' RNA may include a neighbour's: check marker genes before naming their cell type.", C.suggestionFooterInfo], null, ['why']],
  };
  const names = Object.values(LABELS).map(knobName);
  for (const [rule, [suggestion, inView, folded, hint, actions]] of Object.entries(cases)) {
    e.panel.setState({ ...data, suggestion });
    const card = cardOf(e), shown = e.visible(card);
    assert.equal(card.dataset.rule, rule); inOrder(shown, inView);
    for (const text of folded) assert.ok(!shown.includes(text), `${rule}: under Why?: ${text}`);
    assert.ok(cardWords(card) <= 30, `${rule}: ${cardWords(card)} words in view`);
    assert.equal(card.all().find(n => n.className === 'qc-info')?.title ?? null, hint, `${rule}: the Setting line's ⓘ`);
    assert.deepEqual(card.all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), actions, rule);
    e.find('why').click(); inOrder(e.visible(cardOf(e)), [C.why, ...folded]);
    // A Setting line may hold the exact Live knobs name (the deck's one exception, as the harness reads it): only that is
    // taken out before the check; every other word of it, and every tooltip, is checked.
    const texts = card.all().flatMap(n => [n.text, n.title]).filter(Boolean);
    for (const text of texts) assert.deepEqual(bannedWords(settingChecked(text, names)), [], `${rule}: ${text}`);
    assert.ok(!card.textContent.includes('--'), 'never a raw flag');
    assert.ok(!/resolvi/i.test(card.textContent));
    assert.ok(!/Unlikely to help|likely to fix|unlikely to fix/i.test(card.textContent), `${rule}: never that a setting cannot help`);
    assert.doesNotMatch(card.textContent, /for (the whole tissue|areas? )|\bin [A-C]\b|\b[A-C] and [A-C]\b/, `${rule}: no "for area A" line, no letter`);
  }
  // An older engine's drift card ("likely to help the saved results") reads as the rule's own: likely, about 1 in 100.
  e.panel.setState({ ...data, suggestion: { ...cases.R3[0], confidence: 'likelySaved' } });
  assert.ok(e.visible(cardOf(e)).includes(C.likelySize)); assert.ok(!e.visible(cardOf(e)).includes('saved results'));
  // Prior weight's name stands only in its Setting line: U says "this setting", a trial "a firmer hold to the starting
  // cells" (never "Xenium's", whatever the run started from).
  e.panel.setState({ ...data, suggestion: { ...cases.U[0], id: 'U:prior-weight:1:2', flag: 'prior-weight', from: 2, to: 1, changed: { flag: 'prior-weight', from: 1, to: 2 } } });
  inOrder(e.visible(cardOf(e)), ['Like for like, mixing rose from 11.0 to 12.6 in 100 after you changed this setting.', 'Prior weight 2 → 1']);
  assert.equal(cardOf(e).all().find(n => n.className === 'qc-info').title, 'higher = the starting cells hold their transcripts harder');
  // A switch: by what was done ("after you turned one-piece cells off"), its Setting line its checkbox.
  e.panel.setState({ ...data, suggestion: { ...cases.U[0], id: 'U:enforce-connectivity:True:3', flag: 'enforce-connectivity', from: false, to: true,
    changed: { flag: 'enforce-connectivity', from: true, to: false } } });
  assert.ok(cardOf(e).textContent.includes('Like for like, mixing rose from 11.0 to 12.6 in 100 after you turned one-piece cells off.'));
  e.panel.setState({ ...data, suggestion: null, trial: { id: 2, rule: 'R5', area: 'A', flag: 'prior-weight', from: 1, to: 2, start_iter: 300, rows: [] } });
  assert.ok(e.panel.trial.textContent.startsWith('Trying a firmer hold to the starting cells since iteration 300'));
  assert.ok(!/xenium/i.test(e.panel.trial.textContent), 'the hold is never named after Xenium (a start may be Cellpose\'s)');
  e.panel.setState({ ...data, trial: null });
  // U: Undo it and Keep answer it with the card itself; R1 shows the starting outlines of its area.
  e.panel.setState({ ...data, suggestion: cases.U[0] }); e.find('undo-suggestion').click(); e.find('keep-suggestion').click();
  assert.deepEqual(e.events.slice(-2).map(x => [x[0], x[1].rule]), [['onUndo', 'U'], ['onKeep', 'U']]);
  e.panel.setState({ ...data, suggestion: cases.R1[0] }); e.find('show-start').click();
  assert.equal(e.events.at(-1)[0], 'onShowCells'); assert.equal(e.events.at(-1)[2].start, true); assert.equal(e.events.at(-1)[2].area.L, 'B');
  assert.equal(e.events.at(-1)[2].area, data.areas[1], 'its area, outline and all (the page moves the view there)');
  assert.equal(e.find('show-start').title, "Shows the outlines of Xenium's cells and moves the view to where mixing stands out.");
  assert.equal(trialRowText({ kind: 'whole', expected: 9, now: 7 }), 'Whole tissue: 9 → 7 in 100');
});

test('more ideas fold under the card; a chosen idea becomes the card; the fold, Why? and the buttons survive new results', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  const r5 = { id: 'R5:prior-weight:2', rule: 'R5', areas: ['A'], flag: 'prior-weight', from: 1, to: 2, k: 4, confidence: 'rough', timing: 'later', actions: ['new_run', 'try', 'why'], main: 'new_run' };
  suggesting(e, { ...data, suggestion: r4('later'), moreIdeas: [r5] });
  const more = e.panel.suggestions.all().find(n => n.className === 'qc-more');
  // Only what can be picked: R4's untested gentler step is in its Why? text, not under "More ideas".
  assert.ok(more.textContent.startsWith('More ideas (1)')); assert.ok(!more.textContent.includes('gentler')); assert.equal(more.open, false);
  more.open = true; more.emit('toggle');
  const button = e.find('new-run'); e.find('why').click();
  // The next result: new numbers in the same card. Nothing is rebuilt: a press, a tooltip or focus survives.
  const next = state(11.2, 410); next.areas[0].growth = .21;
  suggesting(e, { ...next, suggestion: r4('later', { growth: .21 }), moreIdeas: [r5] });
  assert.equal(e.find('new-run'), button, 'the same button'); assert.ok(cardOf(e).textContent.includes('grew about 21 %'));
  assert.ok(e.visible().includes(C.ruleR4Why), 'Why? stays open'); assert.equal(e.panel.suggestions.all().find(n => n.className === 'qc-more').open, true);
  e.find('idea').click();
  assert.equal(e.events.at(-1)[0], 'onIdea'); assert.equal(cardOf(e).dataset.rule, 'R5');
  assert.ok(!e.visible().includes(C.ruleR4Why), 'another card starts folded');
  assert.ok(e.panel.suggestions.all().find(n => n.className === 'qc-more').textContent.startsWith('More ideas (1)'));
});

test('nothing is clicked twice; New run says why it cannot start; paused, Try it says when a change applies', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  const tryTip = 'Sets Cell compactness to 0.02 for the whole run from the next iteration. Undo puts back 0.04.';
  suggesting(e, { ...data, suggestion: r4('later') });
  const card = cardOf(e), tryButton = e.find('try');
  assert.equal(tryButton.title, tryTip);
  suggesting(e, { ...data, suggestion: r4('later'), acting: true });
  assert.ok(['new-run', 'try'].every(a => e.panel.root.all().find(n => n.dataset.action === a).disabled), 'every action waits while one is on its way');
  assert.equal(e.find('why').disabled, false, 'Why? changes nothing');
  suggesting(e, { ...data, suggestion: r4('later'), canNewRun: false });
  const newRun = e.panel.root.all().find(n => n.dataset.action === 'new-run');
  assert.equal(newRun.disabled, false); assert.equal(newRun.title, C.newRunTooltip);
  // Paused or ended: nothing to pause, so the tooltip does not say it does.
  suggesting(e, { ...data, state: 'paused', suggestion: r4('later') });
  assert.equal(e.find('new-run').title, C.newRunTooltipStill);
  // Paused, Try it's tooltip says that it applies when the run resumes (2026-10-22: no line of its own in view; the trial
  // readout says it once pressed); the card is the same one, not rebuilt.
  suggesting(e, { ...data, state: 'paused', suggestion: r4('later') });
  assert.ok(!cardOf(e).textContent.includes(C.tryPaused)); assert.equal(e.find('try').title, `${tryTip} ${C.tryPaused}`);
  assert.equal(cardOf(e), card, 'pausing changes a tooltip only'); assert.equal(e.find('try'), tryButton);
  suggesting(e, { ...data, suggestion: r4('later') }); assert.equal(e.find('try').title, tryTip, 'running again: as before');
  suggesting(e, { ...data, state: 'scrubbing', suggestion: r4('later') });
  assert.equal(e.panel.suggestions.hidden, true, 'a frame that is not the newest has no card');
  suggesting(e, { ...data, state: 'finished', suggestion: r4('later') });
  assert.deepEqual(cardOf(e).all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), ['new-run', 'why'], 'a run that ended: a new run is all that is left');
  assert.ok(cardOf(e).textContent.includes(C.suggestionFooterEnded)); assert.ok(!cardOf(e).textContent.includes(C.timingLaterNoTime));
  suggesting(e, { ...data, state: 'finished', suggestion: { rule: 'U', areas: [], flag: 'cell-compactness', from: 0.02, to: 0.04, confidence: 'seen' } });
  assert.equal(e.panel.suggestions.hidden, true, 'nothing to put back in a run that ended');
  const early = state(3, 5); early.summary.phase = 'init';
  suggesting(e, { ...early, suggestion: null }); assert.ok(e.visible().includes(C.tooEarly));
  suggesting(e, { ...data, suggestion: null, moreIdeas: [] }); assert.equal(e.panel.suggestions.hidden, true);
});

test('while cells form and nothing is offered the slot says it is too early (R0); a card offered then is shown', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } });
  const forming = state(8, 60); forming.summary.phase = 'burnin';
  suggesting(e, { ...forming, suggestion: null });
  assert.equal(e.panel.suggestions.hidden, false); assert.ok(e.visible().includes(C.tooEarly));
  // Past burn-in but before burn-in + 20 (the start line's areas are shown): cells are still settling.
  suggesting(e, { ...state(8, 110, { areaView: 'start' }), suggestion: null }); assert.ok(e.visible().includes(C.tooEarly));
  const drift = { rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'rough', timing: 'early',
    actions: ['try', 'why'], main: 'try' };
  suggesting(e, { ...forming, suggestion: drift });
  assert.ok(!e.visible().includes(C.tooEarly)); assert.equal(cardOf(e).dataset.rule, 'R3');
  // The switch as its checkbox, its plain change in the line's ⓘ.
  assert.ok(e.visible(cardOf(e)).includes('Untick "Switch off the transcript drift model"')); assert.ok(!e.visible(cardOf(e)).includes('drift on'));
  assert.equal(cardOf(e).all().find(n => n.className === 'qc-info').title, 'drift on');
  assert.equal(e.find('try').title, `Turns drift on for the whole run from the next iteration. Undo puts back drift off. ${C.driftActsLater}`);
  suggesting(e, { ...state(), suggestion: null, moreIdeas: [] });
  assert.equal(e.panel.suggestions.hidden, true, 'cells formed and nothing to suggest: nothing is said');
  // Just undone (the trial's last words on screen): not "too early" beside them; its card comes back once it leads again.
  const undone = { rule: 'R3', flag: 'no-diffusion', from: true, to: false, start_iter: 51, rows: [], verdict: null, outcome: 'undone' };
  suggesting(e, { ...forming, suggestion: null, trialEnded: undone });
  assert.ok(!e.visible().includes(C.tooEarly)); assert.ok(e.panel.trial.textContent.includes('Put back drift off.'));
});

test('a new card slides and fades in once on the stage\'s 150 ms steps (3 paints), then nothing is scheduled', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  suggesting(e, { ...data, suggestion: null });
  suggesting(e, { ...data, suggestion: r4('later') });
  const card = cardOf(e);
  assert.equal(card.style.opacity, String(ENTRANCE[0][0])); assert.equal(card.style.transform, `translateY(${ENTRANCE[0][1]}px)`);
  e.advance(151); assert.equal(card.style.opacity, String(ENTRANCE[1][0]));
  e.advance(150); assert.equal(card.style.opacity, ''); assert.equal(card.style.transform, '');
  assert.equal(e.frames, 2, 'two frames after the result\'s own'); assert.equal(e.tasks.size, 0);
  e.advance(10000); assert.equal(e.frames, 2, 'idle afterwards');
  suggesting(e, { ...state(11.4, 410), suggestion: r4('later', { growth: .2 }) }); assert.equal(e.tasks.size, 0, 'the same card with new numbers stays put');
  const calm = browser(t, { reduced: true, options: { suggestions: true, knobLabel } });
  calm.panel.setState({ ...data, suggestion: null }); calm.panel.setState({ ...data, suggestion: r4('later') });
  assert.equal(calm.tasks.size, 0); assert.equal(cardOf(calm).style.opacity, undefined);
});

test('under Compare (setCovered) nothing in the panel moves: a number counting or a card sliding in lands at once, none starts; uncovered, they move again', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  e.panel.setState(state(8, 400)); e.panel.setState({ ...state(12, 410), motion: true });
  assert.equal(e.tasks.size, 1, 'the number counts');
  e.panel.setCovered(true);
  assert.equal(e.tasks.size, 0, 'nothing scheduled'); assert.equal(e.panel.numberNode.textContent, '12 in 100', 'landed at once');
  const requested = e.requested;
  e.panel.setState({ ...state(20, 420), motion: true });
  assert.equal(e.tasks.size, 0); assert.equal(e.panel.numberNode.textContent, '20 in 100'); assert.equal(e.requested, requested, 'no frame asked for');
  // A card that appears while covered is simply there; one sliding in when Compare opens lands at once.
  suggesting(e, { ...data, suggestion: null }); suggesting(e, { ...data, suggestion: r4('later') });
  assert.equal(e.tasks.size, 0); assert.equal(cardOf(e).style.opacity ?? '', '');
  e.panel.setCovered(false);
  suggesting(e, { ...data, suggestion: null }); suggesting(e, { ...data, suggestion: r4('later') });
  const card = cardOf(e);
  assert.equal(card.style.opacity, String(ENTRANCE[0][0])); assert.equal(e.tasks.size, 1, 'uncovered: it slides in');
  e.panel.setCovered(true);
  assert.deepEqual([e.tasks.size, card.style.opacity, card.style.transform], [0, '', '']);
  e.advance(1000); assert.equal(e.requested, requested, 'still nothing asked for');
  e.panel.setCovered(false); e.panel.setState({ ...state(25, 430), motion: true });
  assert.equal(e.tasks.size, 1, 'uncovered: the number counts again');
});

test('the trial readout: since when, what is watched, expected -> now, the verdict in plain words, Keep and Undo', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } });
  const trial = { id: 3, rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02, asked_iter: 330, start_seq: 34, start_iter: 338,
    rows: [{ kind: 'area', area: 'A', expected: 23, now: 21.4 }, { kind: 'whole', expected: 11.2, now: 10.6 }, { kind: 'depth', expected: 340, now: 309 }],
    verdict: null, notes: [], depth_loss: 9, timing: 'later', later_s: 162 };
  const data = { ...state(10.6, 400), suggestion: r4('later'), trial };
  e.panel.setState(data);
  assert.equal(e.panel.suggestions.hidden, true, 'no card while a trial runs');
  const text = e.panel.trial.textContent;
  inOrder(text, ['Trying tighter shapes since iteration 338', 'Watching where mixing stands out and the whole tissue · 62 iterations so far', C.trialColumns,
    'Where mixing stands out: 23.0 → 21.4 in 100', 'Whole tissue: 11.2 → 10.6 in 100', 'Transcripts per cell: 340 → 309',
    copy('trialPendingLater', { t: '3 min' }), 'Cells also lost 9 % of their transcripts', C.keep, C.undo, C.newRun]);
  const keep = e.find('keep'); keep.click(); e.find('undo').click();
  assert.deepEqual(e.events.slice(-2).map(x => [x[0], x[1].id]), [['onKeep', 3], ['onUndo', 3]]);
  // Its advice for a clear answer, a new run with this setting, is a button beside Keep and Undo (shape knobs).
  e.find('new-run').click(); assert.deepEqual([e.events.at(-1)[0], e.events.at(-1)[1].id], ['onNewRun', 3]);
  assert.deepEqual(e.panel.shown.trial, { id: 3, rule: 'R4', flag: 'cell-compactness', from: 0.04, to: 0.02, start_iter: 338, verdict: null, outcome: null });
  const verdicts = [['less', C.trialLess], ['more', C.trialMore], ['no_change', copy('trialNoChangeAfterLater', { k: '62', t: '3 min' })]];
  for (const [verdict, words] of verdicts) {
    e.panel.setState({ ...data, trial: { ...trial, verdict, rows: trial.rows.map(r => ({ ...r, now: r.now - .1 })) } });
    assert.ok(e.panel.trial.textContent.includes(words), verdict); assert.equal(e.find('keep'), keep, 'Keep survives the results that come');
  }
  assert.equal(trialVerdictText({ ...trial, verdict: 'no_change' }, { k: 480 }), copy('trialNoChangeAfterLaterNoTime', { k: '300' }), 'the verdict window');
  assert.equal(trialVerdictText({ ...trial, timing: 'early', verdict: 'no_change' }, { k: 480 }), 'No clear change after 300 iterations.');
  // In the panel: the forming text names the grey line only while the previous run's line is drawn; once cells have
  // formed it is said in the past (the line has settled; there is just nothing from before the change to compare with).
  const formingTrial = { ...trial, notes: ['forming'], forming: true }, forming = state(8, 60);
  forming.summary.phase = 'burnin';
  const grey = { run: 'r0', series: [{ iter: 0, per100: 9 }, { iter: 500, per100: 11 }] };
  e.panel.setState({ ...forming, previous: null, trial: formingTrial });
  assert.ok(e.panel.trial.textContent.includes(C.trialForming));
  e.panel.setState({ ...forming, previous: grey, trial: formingTrial });
  assert.ok(e.panel.trial.textContent.includes(C.trialForming));
  e.panel.setState({ ...data, previous: null, trial: formingTrial });
  assert.ok(e.panel.trial.textContent.includes(C.trialFormedPast), 'cells formed since: in the past');
  e.panel.setState({ ...data, previous: grey, trial: formingTrial });
  assert.ok(e.panel.trial.textContent.includes(C.trialFormedPast)); assert.ok(!e.panel.trial.textContent.includes('are still forming'));
  assert.equal(trialVerdictText({ ...trial, notes: ['forming'], forming: true }), C.trialForming);
  // A first run of the region has no grey line to point to: the forming text says what is left, a new run.
  assert.equal(trialVerdictText({ ...trial, notes: ['forming'], forming: true }, { previous: false }), C.trialForming);
  assert.ok(!C.trialForming.includes('grey line'));
  assert.equal(trialVerdictText({ ...trial, notes: ['mixed_up'] }), C.trialMixedUp);
  // Drift shows in proseg's count: its trial is judged like any other (an older engine's "no verdict" note says nothing
  // more).
  const drift = { ...trial, rule: 'R3', area: null, flag: 'no-diffusion', from: true, to: false };
  assert.equal(trialVerdictText({ ...drift, verdict: 'less' }), C.trialLess);
  assert.equal(trialVerdictText({ ...drift, notes: ['forming'], forming: true }), C.trialForming);
  assert.equal(trialVerdictText({ ...drift, notes: ['mixed_up'] }), C.trialMixedUp);
  assert.equal(trialVerdictText({ ...drift, notes: ['no_verdict'], verdict: 'more' }), C.trialMore);
  assert.equal(trialVerdictText(drift, { laterTime: '3 min' }), copy('trialPendingLater', { t: '3 min' }));
  // ... and its chart draws the engine's band, with its key, as for any other trial.
  e.panel.setState({ ...data, trial: { ...drift, band: { centre: 10.8, half_width: 1.1 } } });
  assert.ok(e.panel.layout.band.length > 0, 'the band of a drift trial'); assert.ok(e.panel.chartLegend.textContent.includes(C.chartExpected));
  e.panel.setState({ ...data, trial: { ...drift, band: null } }); assert.equal(e.panel.layout.band.length, 0, 'none when the engine has none');
  assert.equal(trialVerdictText({ ...trial, notes: ['no_history'] }), C.trialNoHistory);
  e.panel.setState({ ...data, trial: { ...trial, small_area: true, notes: ['small_area'] } });
  assert.ok(e.panel.trial.textContent.includes(C.trialSmall)); assert.doesNotMatch(e.panel.trial.textContent, /\bArea\b|\barea A\b/);
  e.panel.setState({ ...data, state: 'paused', trial: { ...trial, start_seq: null, start_iter: null, rows: [] } });
  inOrder(e.panel.trial.textContent, ['Trying tighter shapes from the next iteration', C.keep, C.undo, C.tryPaused]);
  assert.ok(!e.panel.trial.textContent.includes('No clear change'), 'nothing is judged before the change is in a frame');
  e.panel.setState({ ...data, state: 'scrubbing', trial: { ...trial, verdict: 'less' } });
  assert.ok(e.panel.trial.textContent.includes(C.trialLess)); assert.equal(e.find('keep'), undefined, 'an earlier frame: read only');
  e.panel.setState({ ...data, state: 'finished' }); assert.ok(e.panel.trial.textContent.includes(C.trialEnded)); assert.equal(e.find('keep'), undefined);
  e.panel.setState({ ...data, trial: null, trialEnded: { ...trial, outcome: 'kept' } });
  assert.ok(e.panel.trial.textContent.includes(C.trialKept)); assert.equal(e.find('keep'), undefined);
  assert.ok(e.panel.trial.textContent.startsWith('Kept tighter shapes since iteration 338'), 'in the past: no "Trying"');
  assert.ok(!e.panel.trial.textContent.includes('so far'), 'nothing counts on once it ended');
  e.panel.setState({ ...data, trial: null, trialEnded: { ...trial, outcome: 'undone', endedIter: 381 } });
  assert.ok(e.panel.trial.textContent.startsWith('Tried tighter shapes from iteration 338 to 381 (undone)'));
  assert.ok(e.panel.trial.textContent.includes('Put back Cell compactness 0.04. Applies from the next iteration.'));
  e.panel.setState({ ...data, trial: null, trialEnded: { ...trial, flag: 'prior-weight', from: 1, to: 2, outcome: 'undone' } });
  assert.ok(e.panel.trial.textContent.includes('Put back the old value, 1. Applies from the next iteration.'), 'Prior weight is named only in a Setting line');
  e.panel.setState({ ...data, trial: null, trialEnded: { ...trial, rule: 'R3', flag: 'no-diffusion', from: true, to: false, outcome: 'undone' } });
  // A switch says what putting it back does, never "Put back Switch off the transcript drift model: on".
  assert.ok(e.panel.trial.textContent.includes('Put back drift off. Applies from the next iteration.'), 'a switch');
  assert.ok(!e.panel.trial.textContent.includes('Switch off'));
  e.panel.setState({ ...data, trial: { ...trial, asked_iter: 330 }, acting: true }); assert.ok(e.panel.trial.all().filter(n => n.tagName === 'BUTTON').every(b => b.disabled));
});

test('a card is known by its id: areas joining or swapping change its words only; a picked idea stays picked', t => {
  const cards = [];
  const e = browser(t, { options: { suggestions: true, knobLabel, onCard: s => cards.push(s) } });
  const data = state(), C2 = { L: 'C', tag: 'rose', per100: 21, start_per100: 14, judged: 320, growth: .12, outline: outline(800, 0) };
  const r2 = areas => ({ id: 'R2:cell-compactness:0.04', rule: 'R2', areas, flag: 'cell-compactness', from: 0.08, to: 0.04, confidence: 'likely',
    timing: 'later', actions: ['new_run', 'try', 'why'], main: 'new_run' });
  const r3 = { id: 'R3:no-diffusion:False', rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'rough', timing: 'later',
    actions: ['try', 'why'], main: 'try' };
  const areas = [...data.areas, C2];
  suggesting(e, { ...data, areas, suggestion: null });
  suggesting(e, { ...data, areas, suggestion: r2(['A']) });
  e.advance(1000);
  e.find('why').click();
  const button = e.find('new-run'), shownCards = cards.length;
  for (const letters of [['A', 'C'], ['C', 'A'], ['A']]) {
    suggesting(e, { ...state(11.1, 410), areas, suggestion: r2(letters) });
    assert.equal(e.find('new-run'), button, `${letters}: the same card`); assert.equal(e.tasks.size, 0, 'no entrance again');
    assert.ok(e.visible().includes(C.ruleR2Why), 'Why? stays open');
  }
  assert.equal(cards.length, shownCards, 'shown once (the harness counts it once)');
  assert.ok(cardOf(e).textContent.includes('cells where mixing stands out grew about 18 %'), 'its why changes with its areas, never a letter');
  assert.ok(!cardOf(e).textContent.includes('for area'));
  // Picked under More ideas: it stays the card while its areas change, and for a moment while a result omits it.
  suggesting(e, { ...state(11, 420), areas, suggestion: r3, moreIdeas: [r2(['A'])] });
  assert.equal(cardOf(e).dataset.rule, 'R3');
  e.find('idea').click(); assert.equal(cardOf(e).dataset.rule, 'R2');
  suggesting(e, { ...state(11.1, 430), areas, suggestion: r3, moreIdeas: [r2(['A', 'C'])] });
  assert.equal(cardOf(e).dataset.rule, 'R2', 'its areas changed: still the one picked');
  suggesting(e, { ...state(11.2, 440), areas, suggestion: r3, moreIdeas: [] });
  assert.equal(cardOf(e).dataset.rule, 'R2', 'one result without it: still shown');
  e.advance(10001);
  suggesting(e, { ...state(11.3, 450), areas, suggestion: r3, moreIdeas: [] });
  assert.equal(cardOf(e).dataset.rule, 'R3', 'gone for longer: the card shown takes its place');
});

test('a card is not rebuilt under a press: the click lands on the card it was aimed at', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  const r3 = { id: 'R3:no-diffusion:False', rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'rough', timing: 'later',
    actions: ['try', 'why'], main: 'try' };
  suggesting(e, { ...data, suggestion: r4('later') });
  const newRun = e.find('new-run');
  e.panel.suggestions.emit('pointerdown');
  suggesting(e, { ...state(11.1, 410), suggestion: r3 });           // another card leads meanwhile
  assert.equal(cardOf(e).dataset.rule, 'R4', 'kept while pressed'); assert.equal(e.find('new-run'), newRun);
  e.panel.suggestions.emit('pointerup'); newRun.click();
  assert.deepEqual([e.events.at(-1)[0], e.events.at(-1)[1].rule], ['onNewRun', 'R4'], 'the click went to the card it was aimed at');
  e.advance(1);
  assert.equal(cardOf(e).dataset.rule, 'R3', 'the change follows once it was let go');
});

test('a run that ended offers a new run for every setting card (drift too); a card landing while paused lands still', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  const r3 = { id: 'R3:no-diffusion:False', rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'rough', timing: 'later',
    actions: ['try', 'why'], main: 'try' };
  suggesting(e, { ...data, state: 'finished', suggestion: r3 });
  assert.deepEqual(cardOf(e).all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), ['new-run', 'why']);
  assert.ok(cardOf(e).textContent.includes(C.suggestionFooterEndedWhole), 'its footer names the button it has');
  assert.equal(e.find('new-run').title, C.newRunTooltipStill);
  // Paused: the paused frame's result brings the first card: it is there, without sliding in.
  const paused = browser(t, { options: { suggestions: true, knobLabel } });
  paused.panel.setState({ ...data, state: 'paused', still: true, motion: false, suggestion: null });
  paused.panel.setState({ ...data, state: 'paused', still: true, motion: false, suggestion: r4('later') });
  paused.advance(1000);
  assert.equal(paused.frames, 0); assert.equal(paused.requested, 0); assert.equal(cardOf(paused).style.opacity, undefined);
});

test('honest words: R2 says what is true of its areas; merged cards give every area\'s numbers; drift is watched like any change', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  const r2 = (extra = {}) => ({ id: 'R2:cell-compactness:0.04', rule: 'R2', areas: ['B'], flag: 'cell-compactness', from: 0.08, to: 0.04, confidence: 'likely',
    timing: 'later', actions: ['new_run', 'try', 'why'], main: 'new_run', ...extra });
  // Area B: already mixed at the start, grown 3 %: "got more mixed" would be false, and so is "reaches further".
  const grown = { ...data, areas: [data.areas[0], { ...data.areas[1], growth: .16 }] };
  suggesting(e, { ...grown, suggestion: r2() });
  const card = cardOf(e);
  assert.ok(e.visible(card).includes('Shapes are looser than standard (0.08); cells where mixing stands out grew about 16 % bigger.'));
  // The gain may be small: under Why? (2026-10-22), the first of its paragraphs.
  assert.ok(!e.visible(card).includes(C.ruleR2NotRose)); e.find('why').click();
  inOrder(e.visible(card), [C.why, C.ruleR2NotRose, C.shapeCost, C.ruleR2Why, C.suggestionFooterNewRun]);
  assert.ok(e.visible(card).includes(C.likelySize));
  suggesting(e, { ...data, suggestion: r2({ areas: ['A'], growth: 0.04 }) });
  const rose = { ...data, areas: [{ ...data.areas[0], growth: .04 }, data.areas[1]] };
  suggesting(e, { ...rose, suggestion: r2({ areas: ['A'], growth: 0.04 }) });
  assert.ok(e.visible(cardOf(e)).includes('Shapes are looser than standard (0.08); cells where mixing stands out got more mixed in this run.'));
  assert.ok(!cardOf(e).textContent.includes(C.ruleR2NotRose)); assert.equal(cardOf(e), card, 'the same card: its words changed in place');
  assert.ok(e.visible(card).includes(C.shapeCost), 'Why? stays open');
  // Several areas: their growth as a range, their squeezed cells added up.
  const two = { ...data, areas: [{ ...data.areas[0], growth: .24, tiny: 9, tiny_start: 1 }, { ...data.areas[1], L: 'C', tag: 'rose', growth: .11, tiny: 4, tiny_start: 1 }] };
  suggesting(e, { ...two, suggestion: r4('later', { areas: ['A', 'C'], growth: .24 }) });
  assert.ok(cardOf(e).textContent.includes('Cells where mixing stands out grew about 11–24 % bigger and seem to reach into their neighbours.'));
  suggesting(e, { ...two, suggestion: { id: 'R5:prior-weight:2', rule: 'R5', areas: ['A', 'C'], flag: 'prior-weight', from: 1, to: 2, k: 8, confidence: 'rough',
    timing: 'later', actions: ['new_run', 'try', 'why'], main: 'new_run' } });
  assert.ok(cardOf(e).textContent.includes('Small cells where mixing stands out are squeezed by their neighbours: 11 have shrunk to almost nothing.'));
  // Drift tried: proseg's count shows it, so its readout watches the whole tissue like any change, its verdict to come.
  const drift = { id: 4, rule: 'R3', area: null, flag: 'no-diffusion', from: true, to: false, start_iter: 380, rows: [], verdict: null, notes: [], timing: 'later' };
  suggesting(e, { ...data, suggestion: null, trial: drift });
  assert.ok(e.panel.trial.textContent.includes(copy('trialWatchingWhole', { k: '20' })), e.panel.trial.textContent);
  assert.ok(e.panel.trial.textContent.includes(C.trialPendingLaterNoTime)); assert.ok(!e.panel.trial.textContent.includes("can't show this change"));
  assert.ok(e.panel.trial.textContent.startsWith('Trying with drift on since iteration 380'));
  assert.deepEqual(e.panel.trial.all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), ['keep', 'undo'], 'drift is no shape knob: Keep and Undo only');
  // An area judged on its own band says so; the last settling stage stops a test that had no verdict.
  const own = { id: 5, rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02, start_iter: 300, rows: [], verdict: 'less', on: 'area', notes: [], timing: 'later' };
  assert.equal(trialVerdictText(own), C.trialLessArea); assert.match(C.trialLessArea, /where mixing stands out/);
  assert.equal(trialVerdictText({ ...own, verdict: 'more' }), C.trialMoreArea);
  assert.equal(trialVerdictText({ ...own, on: 'whole' }), C.trialLess);
  assert.equal(trialVerdictText({ ...own, verdict: null, notes: ['last_stage'] }), C.trialLastStage);
  assert.equal(trialVerdictText({ ...own, verdict: 'less', notes: ['last_stage'] }), C.trialLessArea, 'a verdict reached in time stays');

});

test('every number resting on this kind of run\'s constants says, in its tooltip, they are not yet re-measured on proseg\'s count', t => {
  // The headline's ⓘ said it alone; the arrow (its thresholds), the chip's change and a trial's verdict (its minima and
  // band) rest on the same constants (the summary's constants.measured false: engine/qc_constants.py).
  const e = browser(t, { options: { suggestions: true } }), data = state(11, 400);
  data.summary.change = { matched: -1.6, raw: -2, threshold: 1, model_threshold: 0.6, meaningful: true, direction: 'fewer', enough: true };
  data.summary.constants = { mode: 'region', id: 'outline-2026-10', measured: false };
  e.panel.setState(data);
  assert.ok(e.panel.changeNode.title.endsWith(`\n${C.notRemeasured}`), e.panel.changeNode.title);
  assert.ok(chipModel(data).tooltip.includes(`${C.openPanel} ${C.notRemeasured} `), chipModel(data).tooltip);
  const trial = { id: 5, rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02, start_iter: 300, rows: [], verdict: 'less', on: 'area',
    notes: [], timing: 'later' };
  e.panel.setState({ ...data, trial });
  const verdict = e.panel.trial.all('p').find(p => p.textContent === C.trialLessArea);
  assert.equal(verdict.title, C.notRemeasured);
  // Re-measured (a new constants id): none of them says it; nor does a line with no verdict.
  const measured = { ...data, summary: { ...data.summary, constants: { ...data.summary.constants, measured: true } } };
  e.panel.setState({ ...measured, trial });
  assert.ok(!e.panel.changeNode.title.includes(C.notRemeasured)); assert.ok(!chipModel(measured).tooltip.includes(C.notRemeasured));
  assert.equal(e.panel.trial.all('p').find(p => p.textContent === C.trialLessArea).title, '');
  e.panel.setState({ ...data, trial: { ...trial, verdict: null } });
  assert.ok(e.panel.trial.all('p').every(p => !(p.title ?? '').includes(C.notRemeasured)));
  // No change after its iterations is a verdict on the same band; a note's line (too few to tell) is none.
  const titleOf = text => e.panel.trial.all('p').find(p => p.textContent === text)?.title;
  e.panel.setState({ ...data, trial: { ...trial, verdict: 'no_change', iterations: 40 } });
  assert.equal(titleOf(copy('trialNoChangeAfterLaterNoTime', { k: '40' })), `${C.notRemeasured}\n${C.outlineEraTooltip}`, 'its timing figure: the outline-era test (H5)');
  e.panel.setState({ ...data, trial: { ...trial, notes: ['no_history'] } });
  assert.equal(titleOf(C.trialNoHistory), '');
  assert.equal(trialVerdictReached({ ...trial, verdict: 'none', notes: ['last_stage'] }), false);
  assert.equal(trialVerdictReached({ ...trial, verdict: 'more' }), true); assert.equal(trialVerdictReached(null), false);
});

test('actions are callbacks with the original objects; the switch never opens or closes the panel; it waits for the answer', t => {
  const e = browser(t), data = state(); e.panel.setState(data);
  const clicked = e.find('toggle'), event = { prevented: 0, preventDefault() { this.prevented++; }, stopPropagation() {} };
  clicked.onclick(event); assert.equal(event.prevented, 1, 'a click in the summary does not toggle the details');
  e.find('show-cells').click(); assert.equal(e.find('area'), undefined, 'no area row to click (2026-10-22)');
  assert.deepEqual(e.events.map(x => x[0]), ['onToggle', 'onShowCells']);
  assert.equal(e.events[0][1], false); assert.equal(e.events[1][1], true);
  e.panel.setState({ ...data, showingCells: true });
  assert.equal(e.panel.showCells.getAttribute('aria-pressed'), 'true'); e.find('show-cells').click(); assert.equal(e.events.at(-1)[1], false);
  e.panel.setState({ ...data, enabled: false, state: 'off' }); e.find('toggle').click(); assert.deepEqual(e.events.at(-1), ['onToggle', true]);
  assert.equal(e.panel.toggle.textContent, C.offLabel); assert.equal(e.panel.toggle.getAttribute('aria-checked'), 'false');
  e.panel.setState({ state: 'loading' }); assert.equal(e.panel.toggle.disabled, true, 'no switch before the engine said how it is');
  e.panel.setState({ ...data, state: 'failed', reason: 'timeout' }); e.find('retry').click(); assert.deepEqual(e.events.at(-1), ['onRetry']);
  assert.equal(e.panel.toggle.getAttribute('role'), 'switch');
});

test('nothing is scheduled on mount, for a repeated state, or for a change without state.motion (a crop\'s ten a second)', t => {
  const e = browser(t), data = state(8.2); e.panel.setState(data);
  assert.equal(e.requested, 0); assert.equal(e.tasks.size, 0);
  const clears = e.clearCount(), nodes = e.panel.howBody.childNodes;
  e.panel.setState(JSON.parse(JSON.stringify(data))); e.advance(10000);
  assert.equal(e.clearCount(), clears); assert.equal(e.panel.howBody.childNodes, nodes, 'nothing rebuilt');
  assert.equal(e.requested, 0); assert.equal(e.tasks.size, 0);
  e.panel.setState(state(12, 410)); assert.equal(e.panel.numberNode.textContent, '12 in 100', 'lands in the snapshot\'s own frame');
  assert.equal(e.tasks.size, 0); assert.equal(e.requested, 0); assert.ok(e.clearCount() > clears, 'drawn once, now');
});

test("with state.motion a changed number counts for 450 ms and the newest segment draws on, on the stage's 150 ms steps", t => {
  const e = browser(t); e.panel.setState(state(8, 400)); e.panel.setState({ ...state(12, 410), motion: true });
  assert.equal(e.panel.numberNode.textContent, '8 in 100'); assert.equal(e.tasks.size, 1);
  e.advance(101); assert.equal(e.panel.displayed, 8, 'nothing between the steps the stage paints anyway');
  e.advance(50); assert.ok(e.panel.displayed > 8 && e.panel.displayed < 12, 'the first step at 150 ms');
  const times = []; const paint = e.panel.paintMotion.bind(e.panel); e.panel.paintMotion = ms => { times.push(Math.round(e.now)); paint(ms); };
  e.advance(400); assert.equal(e.panel.numberNode.textContent, '12 in 100'); assert.equal(e.tasks.size, 0);
  assert.deepEqual(times, [301, 451], 'the cell tint and the areas step at 150, 300 and 450 ms too (+ the frame)');
  assert.ok(e.requested <= 3 && e.frames === e.requested, `${e.requested} frames, ${e.frames} counted`);
  const clears = e.clearCount(), frames = e.requested;
  e.advance(10000); assert.equal(e.clearCount(), clears); assert.equal(e.requested, frames, 'idle afterwards');
  e.panel.setState({ ...state(15, 420), motion: true }); e.advance(151);
  const from = e.panel.displayed; e.panel.setState({ ...state(20, 430), motion: true }); assert.equal(e.panel.motion.from, from, 'retargets from what is shown');
  e.panel.setState({ ...state(5), runId: 'run-b', motion: true }); assert.equal(e.panel.numberNode.textContent, '5 in 100'); assert.equal(e.tasks.size, 0);
});

test('reduced motion, a collapsed panel or a hidden page stay still; disposal cancels everything', t => {
  const e = browser(t, { reduced: true }); e.panel.setState(state(8)); e.panel.setState({ ...state(12, 410), motion: true });
  assert.equal(e.panel.numberNode.textContent, '12 in 100'); assert.equal(e.requested, 0); assert.equal(e.tasks.size, 0);
  e.media.matches = false; e.media.emit('change'); e.panel.setState({ ...state(20, 420), motion: true }); assert.equal(e.tasks.size, 1);
  e.media.matches = true; e.media.emit('change'); assert.equal(e.tasks.size, 0); assert.equal(e.panel.numberNode.textContent, '20 in 100');
  e.media.matches = false;
  e.panel.root.open = false; e.panel.root.emit('toggle'); const count = e.clearCount();
  e.panel.setState({ ...state(18, 440), motion: true }); assert.equal(e.tasks.size, 0); assert.equal(e.clearCount(), count, 'a collapsed chart is not drawn');
  e.panel.open(); assert.equal(e.tasks.size, 0); assert.ok(e.clearCount() > count); assert.deepEqual(e.scrolled, { block: 'nearest', behavior: 'auto' });
  e.panel.setState({ ...state(8, 450), motion: true });
  e.document.hidden = true; for (const fn of e.document.listeners.get('visibilitychange')) fn(); assert.equal(e.tasks.size, 0);
  e.document.hidden = false; e.panel.setState({ ...state(25, 460), motion: true }); assert.equal(e.tasks.size, 1);
  e.panel.dispose(); assert.equal(e.tasks.size, 0); assert.equal(e.observers[0].disconnected, true);
  assert.equal(e.panel.root.childNodes.length, 0, 'the page\'s own <details> is emptied, not removed');
  assert.equal([...e.media.listeners.values()].flat().length, 0);
});

test('the chart follows its box from the observer, hovers by the pointer\'s own offset (no layout read) and by keys', t => {
  const e = browser(t); e.panel.setState(state());
  assert.equal(e.panel.layout.width, 218); assert.equal(e.panel.layout.height, 132);
  e.observers[0].fn([{ contentRect: { width: 260, height: 400 } }]); assert.equal(e.panel.layout.width, 260);
  e.layoutReads = 0;
  e.panel.canvas.emit('pointermove', { offsetX: e.panel.layout.x(400) });
  assert.ok(e.panel.readout.textContent.startsWith('Iteration 400 · 11.0 in 100 cells look mixed (of 3,188 with 100 or more transcripts)'));
  assert.equal(e.layoutReads, 0);
  e.panel.canvas.emit('pointermove', { offsetX: e.panel.layout.x(350) });
  assert.ok(e.panel.readout.textContent.endsWith('· after standard shapes, iteration 300'));
  e.panel.canvas.emit('keydown', { key: 'Home' }); assert.ok(e.panel.readout.textContent.includes('Iteration 20'));
  e.panel.canvas.emit('keydown', { key: 'Escape' }); assert.equal(e.panel.readout.hidden, true);
  assert.equal(e.requested, 0);
});

test("the chart's start line is named as what it is, proseg's count at iteration 0, as the line above it says, whatever the cells the run started from", t => {
  // H3: the start line was named after its cells ("Xenium's cells 11"); it is proseg's count at iteration 0.
  const e = browser(t), drawn = () => e.calls.filter(c => c[1] === 'fillText').map(c => String(c[2]));
  e.panel.setState(state());
  assert.equal(e.panel.layout.start.label, 'Iteration 0: 10'); assert.ok(e.panel.chartTexts().includes('Iteration 0: 10'));
  assert.ok(e.panel.startNode.title.startsWith("At the start (proseg's count at iteration 0): 10 in 100."), e.panel.startNode.title);
  // A run from another segmentation (a CosMx sample's cells, Cellpose's): the same words, never Xenium's.
  e.calls.length = 0;
  e.panel.setState(state(11, 400, { startSource: 'other', seriesKey: 401 }));
  assert.equal(e.panel.layout.start.label, 'Iteration 0: 10');
  assert.ok(e.panel.chartTexts().includes('Iteration 0: 10')); assert.ok(drawn().includes('Iteration 0: 10'), drawn().join(' | '));
  assert.ok(![...e.panel.chartTexts(), ...drawn(), e.panel.startNode.title].some(text => /Xenium|starting cells/.test(text)), 'never named after its cells');
  // A start from nuclei only has no start line at all.
  e.panel.setState(state(11, 400, { startSource: 'nuclei', start: null }));
  assert.equal(e.panel.layout.start, null); assert.ok(!e.panel.chartTexts().some(text => /Iteration 0|cells \d/.test(text)));
  e.panel.setState(state());
  assert.equal(e.panel.layout.start.label, 'Iteration 0: 10');
});

test('"How is this estimated?" is collapsed: five short points naming ResolVI, how the model was learned, then the method\'s authors; never the user\'s files', t => {
  const e = browser(t);
  const own = { table: 'xenium_table', other_table: 'table', priors: 'pooled', epochs_run: 50, epochs_notebook: 50, cells: 15000, notebook_equivalent: true, differences: [] };
  e.panel.setState(state(11, 400, { sampleModel: { ...state().sampleModel, table: 'xenium_table', provenance: own } }));
  assert.equal(e.panel.how.open, false);
  const items = () => e.panel.how.all().filter(n => n.tagName === 'LI').map(n => n.textContent);
  const learned = () => e.panel.how.all().find(n => n.className === 'qc-how-learned');
  assert.deepEqual(items(), [C.how1, C.how2, C.how3, C.how4, C.how5]);
  assert.equal(learned().textContent, 'Learned from 15,000 cells on 17 Oct 2026, in 2 min.'); assert.equal(learned().title, "The cells it learned from: Xenium's cells.");
  const credit = e.panel.how.all().find(n => n.className === 'qc-how-credit');
  assert.equal(credit.textContent, C.howCredit, 'whose method it is, last');
  const words = e.panel.how.all().filter(n => ['LI', 'P'].includes(n.tagName) && n !== credit).map(n => n.textContent).join(' ').split(/\s+/).length;
  assert.ok(words <= 90, `${words} words`);
  // Outside it the method stands in the panel's title alone.
  const outside = e.panel.root.all().filter(n => !e.panel.how.contains(n) && n.text).map(n => n.text);
  assert.deepEqual(outside.filter(text => /resolvi/i.test(text)), [C.title]);
  // A small table learned with more passes: the number in the line (more_epochs_small_table); nothing
  // compares the model with anything of the user's, the same model or not.
  const small = { ...own, table: 'table', priors: 'single', epochs_run: 429, notebook_equivalent: false, differences: ['more_epochs_small_table', 'single_table_priors'] };
  e.panel.setState(state(11, 400, { sampleModel: { ...state().sampleModel, table: 'xenium_table', provenance: small } }));
  assert.equal(learned().textContent, 'Learned in 429 passes over 15,000 cells on 17 Oct 2026, in 2 min.');
  assert.equal(learned().title, `The cells it learned from: the sample's other cell table. ${C.howPasses}`, 'the table in plain words, never its zarr key');
  // The summary's own provenance wins; a model learned before provenance was recorded: the plain line.
  e.panel.setState(state(11, 400, { summary: { ...state().summary, provenance: own }, sampleModel: { ...state().sampleModel, provenance: small } }));
  assert.equal(learned().textContent, 'Learned from 15,000 cells on 17 Oct 2026, in 2 min.');
  e.panel.setState(state(11, 400, { sampleModel: { ...state().sampleModel, table: 'xenium_table', provenance: null } }));
  assert.equal(learned().textContent, 'Learned from 15,000 cells on 17 Oct 2026, in 2 min.');
  e.panel.setState(state(11, 400, { sampleModel: null })); assert.equal(learned(), undefined, 'no model: no line');
  for (const provenance of [own, small, null]) {
    e.panel.setState(state(11, 400, { sampleModel: { ...state().sampleModel, table: 'xenium_table', provenance } }));
    const all = e.panel.root.all().flatMap(n => [n.text, n.title, n.getAttribute?.('aria-label')]).filter(Boolean).join('\n');
    assert.doesNotMatch(all, /notebook|part\s*3|same (settings|model)|xenium_table/i);
  }
});

test('every string the panel and the chip can show (text, titles, labels) is free of banned words, in every state', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } });
  const chipHost = e.document.createElement('div'), chip = new QcChip(chipHost);
  const seen = new Set();
  const collect = root => { for (const n of root.all()) { for (const s of [n.text, n.title, n.getAttribute?.('aria-label')]) if (s) seen.add(s); } };
  const variants = [];
  for (const status of [...PANEL_STATES, 'loading']) for (const reason of [null, 'queue', 'lock', 'gpu_busy', 'gpu_memory', 'gpu', 'too_few_cells', 'unreadable',
    'oom', 'not_settled', 'timeout', 'unexpected', 'no_table', 'not_counts', 'software', 'no_transcripts', 'analysis', 'no_gpu',
    'torch_cpu', 'driver', 'no_counts', 'old_proseg']) {
    variants.push({ ...state(), state: status, reason, progress: { step: 10, of: 50 }, eta_s: 70, shared: true, relearn: reason === 'queue', requested: true,
      prosegFix: ['build', 'install', 'setting'][variants.length % 3] });
    if (status === 'off') variants.push({ ...state(), state: 'off', enabled: false, sampleState: 'unavailable', sampleReason: reason });
  }
  variants.push({ ...state(), state: 'paused', atStart: true, summary: null }, { ...state(), tiled: true, overview: true, view3d: true },
    { ...state(), restarting: true }, { ...state(), startSource: 'nuclei', start: null }, { ...state(), startSource: 'other' },
    // Proseg's own count: a start with nuclei, a result of another iteration looked back at, thresholds not re-measured.
    { ...state(), startKind: 'paired' }, { ...state(), state: 'paused', atStart: true, startKind: 'paired', summary: null },
    { ...state(), state: 'scrubbing', snapped: 390, frameIter: 401 }, { ...state(), state: 'paused', snapped: 390, frameIter: 401 },
    { ...state(), summary: { ...state().summary, constants: { mode: 'tiled', id: 'outline-2026-10', measured: false } } },
    { ...state(), state: 'finished', runState: 'done', startKind: 'paired' },
    { ...state(), suggestion: { rule: 'R4', areas: ['A'], from: .04, to: .02, timing: 'later' }, moreIdeas: [{ rule: 'R2', areas: ['A'], from: .08, to: .04 }] },
    { ...state(), motionBudgetExceeded: true });
  // Every card, the trial readout in each of its states, the comparison with the previous run.
  const trial = { id: 1, rule: 'R4', area: 'A', flag: 'cell-compactness', from: .04, to: .02, start_iter: 338, asked_iter: 330, verdict: null, notes: [],
    rows: [{ kind: 'area', area: 'A', expected: 23, now: 21.4 }, { kind: 'whole', expected: 11.2, now: 10.6 }, { kind: 'depth', expected: 340, now: 309 }], depth_loss: 9, later_s: 162 };
  const cards = [r4('early'), r4('later', { later_s: 162 }),
    { rule: 'U', areas: ['A'], flag: 'prior-weight', from: 2, to: 1, changed: { flag: 'prior-weight', from: 1, to: 2 }, before_per100: 11, after_per100: 12.4, confidence: 'seen' },
    { rule: 'R1', areas: ['B'], confidence: 'unlikely' }, { rule: 'R2', areas: [], flag: 'cell-compactness', from: .08, to: .04, v: .08, confidence: 'likely', timing: 'early' },
    { rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'rough', actions: ['try', 'why'], main: 'try' },
    { rule: 'R5', areas: ['A'], flag: 'prior-weight', from: 1, to: 2, k: 4, confidence: 'rough', timing: 'later' },
    { rule: 'R6', areas: ['A'], confidence: 'unlikely' }, { rule: 'R7', areas: ['A'], tried: { flag: 'cell-compactness', from: .04, to: .02 }, confidence: 'unlikely' },
    // This round's engine: "unknownCause", and the inspect action.
    { rule: 'R1', areas: ['B'], confidence: 'unknownCause', actions: ['show_start', 'inspect'], main: 'show_start' },
    { rule: 'R6', areas: ['A'], confidence: 'unknownCause', actions: ['inspect'], main: 'inspect' },
    { rule: 'R7', areas: ['A'], tried: { flag: 'cell-compactness', from: .04, to: .02 }, confidence: 'seen', actions: ['inspect'], main: 'inspect' }];
  for (const suggestion of cards) for (const status of ['live', 'paused', 'finished']) variants.push({ ...state(), state: status, suggestion, moreIdeas: cards.slice(0, 3) });
  for (const extra of [{}, { verdict: 'less' }, { verdict: 'more' }, { verdict: 'no_change' }, { notes: ['forming'], forming: true }, { notes: ['mixed_up'] }, { notes: ['no_matched'] },
    { rule: 'R3', flag: 'no-diffusion', from: true, to: false }, { notes: ['no_history'] }, { notes: ['small_area'], small_area: true },
    { start_iter: null, start_seq: null, rows: [] }]) {
    variants.push({ ...state(), trial: { ...trial, ...extra } }, { ...state(), state: 'paused', trial: { ...trial, ...extra } });
  }
  for (const trialEnded of [{ ...trial, outcome: 'kept' }, { ...trial, outcome: 'undone' }, { ...trial, flag: 'prior-weight', from: 1, to: 2, outcome: 'undone' }])
    variants.push({ ...state(), trialEnded });
  const previous = { run: 'r0', series: [{ iter: 0, per100: 9 }, { iter: 800, per100: 12.5 }] };
  for (const verdict of ['less', 'same', 'more', null]) {
    const v = { ...state(), previous }; v.summary = { ...v.summary, previous: { run: 'r0', iter: 400, this: 11, previous: 12.4, verdict, spread: .5, clearly: 1 } };
    variants.push(v);
  }
  // The like-for-like change and the previous run's, both ways and light; coverage (too few, most cells too
  // small), retention (kept, a loss, a gain, about the same); areas judged on their own, at the edge, already mixed;
  // the model's provenance (the notebook's own, what differs, not recorded).
  const changes = [{ matched: -1.8, threshold: .7, meaningful: true }, { matched: 1.3, threshold: .7, meaningful: true }, { matched: .4, threshold: .3, meaningful: true },
    { matched: -.4, threshold: .3, meaningful: true }, { matched: .1, threshold: .7, meaningful: false }];
  for (const change of changes) {
    const v = { ...state(), previous }; v.summary = { ...v.summary, change, prev_change: change, previous: { run: 'r0', iter: 400, this: 11, previous: 12.4 } };
    variants.push(v, { ...v, state: 'scrubbing' });
  }
  for (const [coverage, retention] of [[{ cells: 3, judged: 1, empty: 2, under20: 0, under100: 0 }, null], [{ cells: 8617, judged: 3100 }, { ratio: .7 }],
    [{ cells: 3439, judged: 3188 }, { ratio: .95 }], [{ cells: 3439, judged: 3188 }, { ratio: 1.2 }], [{ cells: 3439, judged: 3188 }, { ratio: 1.001 }]]) {
    const v = state(); v.summary = { ...v.summary, coverage, retention, judged: coverage.judged }; variants.push(v);
  }
  const localAreas = [{ ...state().areas[0], edge: true, local: { judged: 420, paired: 410, matched_change: -2.6, threshold: 2.1, enough: true, direction: 'fewer' } },
    { ...state().areas[1], edge: true, local: { judged: 220, paired: 200, matched_change: 1, threshold: 2.8, enough: false, direction: 'same' } },
    { ...state().areas[1], L: 'C', local: { judged: 500, paired: 480, matched_change: 0.4, threshold: 1.9, enough: true, direction: 'same' } },
    { ...state().areas[0], L: 'D', local: { judged: 500, paired: 480, matched_change: 3.1, threshold: 1.9, enough: true, direction: 'more' } }];
  variants.push({ ...state(), areas: localAreas });
  const provenances = [null, { table: 'xenium_table', notebook_equivalent: true, differences: [] },
    { table: 'table', epochs_run: 429, epochs_notebook: 50, notebook_equivalent: false, differences: ['more_epochs_small_table', 'single_table_priors', 'other_table_used', 'x'] }];
  for (const provenance of provenances) variants.push({ ...state(), sampleModel: { ...state().sampleModel, table: 'xenium_table', provenance } });
  // The panel's own "How is this estimated?" names the method: the only exception (its words in every variant).
  const how = new Set();
  for (const v of variants) {
    e.panel.setState(v); chip.setState(v); collect(e.panel.root); collect(chipHost);
    for (const n of e.panel.how.all()) if (n.text) how.add(n.text);
    if (e.find('why')) { e.find('why').click(); collect(e.panel.root); }   // the Why? texts
  }
  for (const text of seen) {
    const allowed = how.has(text) ? text.replace(/ResolVI/g, '') : text;
    // A Setting line's exact Live knobs name is the one exception: only it is taken out, the rest of the line is checked.
    assert.deepEqual(bannedWords(settingChecked(allowed, Object.values(LABELS).map(knobName))), [], text);
    if (!how.has(text) && text !== C.title) assert.ok(!/resolvi/i.test(text), text);
  }
  assert.ok(seen.size > 150, `${seen.size} strings checked`);
});

test('the chip: number, direction, a 48 by 14 sparkline, its states, hidden when off or unavailable, a click', t => {
  const e = browser(t), host = e.document.createElement('div'), clicks = [];
  const chip = new QcChip(host, { onClick: s => clicks.push(s) }); const data = state();
  data.summary.change = { matched: 1.24, threshold: 0.7, model_threshold: 0.6, meaningful: true, direction: 'more' }; chip.setState(data);
  // "like for like" beside the raw number.
  assert.equal(chip.label.textContent, 'Mixing 11 in 100 · ▲1.2 like for like'); assert.equal(host.hidden, false);
  // Its tooltip: from which iteration a live number is smoothed (chipSmoothed, H10), and which count it reads.
  assert.equal(chip.root.title, `11 in 100 cells look mixed; 1.2 more than at the start, like for like. Click to open the mixing check. ${copy('chipSmoothed', { i: '400' })} ${C.chipCountAveraged}`);
  assert.equal(chip.canvas.style.width, '48px'); assert.equal(chip.canvas.style.height, '14px');
  assert.equal(chip.canvas.width, 96); assert.equal(chip.canvas.height, 28);
  chip.root.click(); assert.equal(clicks[0], data);
  const held = state(8.7); held.summary.shown = 8; held.summary.change = { matched: -1.71, threshold: 0.7, meaningful: true }; chip.setState(held);
  // Without the change's own model threshold, a region's (MODEL_THRESHOLD, 2.3 on proseg's count): 1.7 is drawn hollow.
  assert.equal(chip.label.textContent, 'Mixing 8 in 100 · ▽1.7 like for like', "the engine's integer, the like-for-like change");
  // 12.49 at the start, 12.51 now once read "Mixing 13 in 100 ▲1"; the engine's change is under its threshold.
  const repro = state(12.51); repro.start = { ...repro.start, per100: 12.49 }; repro.summary.shown = 13;
  repro.summary.change = { raw: 0.02, matched: 0.02, threshold: 0.21, meaningful: false, direction: 'same' };
  assert.equal(chipModel(repro).text, 'Mixing 13 in 100 · same as at the start', '"like for like" in its tooltip');
  assert.match(chipModel(repro).tooltip, /about the same as at the start, like for like\./);
  assert.equal(chipModel({ ...repro, summary: { ...repro.summary, change: null } }).text, 'Mixing 13 in 100', 'no like-for-like change: no direction');
  const light = state(); light.summary.change = { matched: -0.45, threshold: 0.3, model_threshold: 0.6, meaningful: true, direction: 'fewer' };
  assert.equal(chipModel(light).text, 'Mixing 11 in 100 · ▽0.5 like for like', 'a hollow arrow under the model\'s own threshold');
  // A held verdict (the engine's Steady): its own word, though the change just crossed back under the threshold.
  const steady = state(); steady.summary.change = { matched: -0.97, threshold: 1.0, model_threshold: 0.6, meaningful: true, direction: 'fewer', direction_now: 'same', held: true };
  assert.equal(chipModel(steady).text, 'Mixing 11 in 100 · ▼1.0 like for like');
  // ... and its tooltip says the word is held and what the change itself reads now (H2), beside the smoothing.
  assert.equal(chipModel(steady).tooltip, `11 in 100 cells look mixed; 1.0 fewer than at the start, like for like. Click to open the mixing check. ${copy('chipSmoothed', { i: '400' })} `
    + `This word is held a few seconds so it doesn't flicker; the change itself now reads about the same as at the start. ${C.chipCountAveraged}`);
  assert.ok(!chipModel({ ...steady, summary: { ...steady.summary, change: { ...steady.summary.change, held: false } } }).tooltip.includes('held a few seconds'), 'unheld: nothing');
  // F8: a held word beside the size it was decided on (the held block's per100_now and per100_start, those its word
  // compared), never the change's own size now (0.3, inside the threshold): in the chip's text and its tooltip alike.
  const decided = state(); decided.summary.change = { matched: -0.3, threshold: 0.7, model_threshold: 2.3, meaningful: true, direction: 'fewer',
    direction_now: 'same', held: true, per100_now: 8.6, per100_start: 11.1 };
  assert.equal(chipModel(decided).text, 'Mixing 11 in 100 · ▼2.5 like for like');
  assert.equal(chipModel(decided).tooltip, `11 in 100 cells look mixed; 2.5 fewer than at the start, like for like. Click to open the mixing check. ${copy('chipSmoothed', { i: '400' })} `
    + `This word is held a few seconds so it doesn't flicker; the change itself now reads about the same as at the start. ${C.chipCountAveraged}`);
  chip.setState(decided); assert.equal(chip.label.textContent, 'Mixing 11 in 100 · ▼2.5 like for like');
  e.panel.setState(decided); assert.equal(e.panel.changeNode.textContent, '▼ Fewer, like for like', 'the line\'s solid arrow, the same size');
  // Under 300 judged cells its tooltip says the number is rough.
  const rough = state(); rough.summary.judged = 120;
  assert.ok(chipModel(rough).tooltip.startsWith(C.roughNumber), chipModel(rough).tooltip);
  // E2: a hollow arrow holds for this model only, said so (changeLightTooltip's clause).
  assert.equal(chipModel(light).tooltip, `11 in 100 cells look mixed; 0.5 fewer than at the start, like for like. ${C.changeLightTooltip} Click to open the mixing check. `
    + `${copy('chipSmoothed', { i: '400' })} ${C.chipCountAveraged}`);
  // A saved iteration (count "sampled": the engine's single result on the count proseg saves there) and an own number
  // (single: a resting run's newest) are not smoothed; the saved one reads the count proseg saves.
  // With its change beside it, each part's count is named (F3): the number the count proseg saves, the change the
  // averaged count, like for like; without a change only the number's.
  const saved = state(); saved.summary = { ...saved.summary, single: true, count: 'sampled', change: light.summary.change };
  assert.ok(chipModel(saved).tooltip.endsWith(`Click to open the mixing check. ${C.chipCountSavedChange}`), chipModel(saved).tooltip);
  assert.ok(!chipModel(saved).tooltip.includes(C.chipCountSaved), 'never "Read on the count proseg saves" beside a change read on the averaged count');
  assert.equal(C.chipCountSavedChange, "The number reads the count proseg saves at this iteration; the change since the start reads proseg's count averaged over its random background draw, on both sides.");
  const savedAlone = state(); savedAlone.summary = { ...savedAlone.summary, single: true, count: 'sampled', change: null };
  assert.equal(chipModel(savedAlone).text, 'Mixing 11 in 100');
  assert.ok(chipModel(savedAlone).tooltip.endsWith(C.chipCountSaved), chipModel(savedAlone).tooltip);
  const savedForming = state(7, 40); savedForming.summary = { ...savedForming.summary, phase: 'burnin', single: true, count: 'sampled', change: light.summary.change };
  assert.ok(chipModel(savedForming).tooltip.endsWith(C.chipCountSaved), 'cells forming: no change beside the number');
  const own = state(); own.summary = { ...own.summary, single: true, count: 'averaged' };
  assert.ok(chipModel(own).tooltip.endsWith(`Click to open the mixing check. ${C.chipCountAveraged}`), chipModel(own).tooltip);
  assert.ok(!chipModel(own).tooltip.includes('smoothed'));
  // The chip writes its tooltip again when only the count or single changes (a frame counted again for the hold: the
  // same numbers, read on the count proseg saves; a resting run's own number), never keeping the words of before.
  const recount = state(); chip.setState(recount);
  assert.ok(chip.root.title.endsWith(C.chipCountAveraged), chip.root.title);
  chip.setState({ ...recount, summary: { ...recount.summary, count: 'sampled' } });
  assert.ok(chip.root.title.endsWith(C.chipCountSaved), chip.root.title);
  chip.setState({ ...recount, summary: { ...recount.summary, count: 'sampled', single: true } });
  assert.ok(!chip.root.title.includes('smoothed'), chip.root.title);
  assert.equal(chip.root.getAttribute('aria-label'), `${chip.label.textContent}. ${chip.root.title}`);
  // Too few judged cells for a number.
  const few = state(); few.summary.judged = 1; assert.equal(chipModel(few).text, C.chipTooFew); assert.equal(chipModel(few).shown, null);
  chip.setState({ ...state(), state: 'learning', progress: { step: 20, of: 50 } }); assert.equal(chip.label.textContent, 'Mixing check: learning this sample… 40 %');
  chip.setState({ ...state(), state: 'waiting' }); assert.equal(chip.label.textContent, C.chipWaiting); assert.equal(chip.canvas.hidden, true);
  for (const status of ['off', 'unavailable', 'loading']) { chip.setState({ ...state(), state: status }); assert.equal(host.hidden, true, status); }
  chip.setState({ ...state(), enabled: false }); assert.equal(host.hidden, true);
  chip.setState({ ...state(), state: 'paused', atStart: true, summary: null }); assert.equal(chip.label.textContent, 'Mixing 10 in 100');
  assert.equal(chipModel({ ...state(8), startSource: 'nuclei' }).text, 'Mixing 8 in 100');
  assert.equal(chipModel({ ...state(), summary: null }).text, C.chipFrame);
  assert.equal(chipModel({ ...state(), state: 'scrubbing' }).tooltip.startsWith('Iteration 400 (not the newest).'), true);
  assert.equal(e.requested, 0); chip.dispose(); assert.equal(chip.root.parentElement, null);
});

test('fromView: the panel gets what qc-client.mjs decided and what the page knows, nothing else', () => {
  assert.deepEqual(fromView(null), { state: 'loading' });
  const view = { status: 'live', enabled: true, runId: 'r', model: 'm', summary: { seq: 3 }, start: { per100: 9 }, startSource: 'xenium', atStart: false,
    reason: null, progress: null, eta_s: null, shared: false, relearn: false, requested: false, restarting: false, line: [{ seq: 3 }], lineVersion: 7,
    ticks: [], formingUntil: 100, frameIter: 30, areas: [], batch: 12, sampleModel: null, tiled: true, motion: false,
    cells: new Uint8Array(40000) };
  const s = fromView(view, { sampleName: 'sample-A', showingCells: true });
  assert.equal(s.state, 'live'); assert.equal(s.series, view.line); assert.equal(s.seriesKey, 7); assert.equal(s.sampleName, 'sample-A');
  assert.equal('cells' in s, false, 'per-cell values never reach the panel (its change key would hold 40,000 numbers)');
  // No number from outside the app (the view has none to give, nor does the panel take one).
  assert.equal('notebook' in fromView({ ...view, notebook: 12.9 }), false);
  // 2026-10-22: the engine's areas for the cards (and area_view: cells forming while "start"), nothing for an area list.
  const listed = fromView({ ...view, areas: [{ L: 'A' }], areaView: 'start', startAreas: true, moreSpots: 3, listed: 3, areasKnown: true, areasFrom: 120 });
  assert.deepEqual([listed.areas, listed.areaView], [[{ L: 'A' }], 'start']);
  for (const key of ['startAreas', 'moreSpots', 'listed', 'areasKnown', 'areasFrom']) assert.equal(key in listed, false, key);
});

test('standalone demo initializes and drives every state without a browser or background updates', async t => {
  // The demo is a development page: a deploy leaves frontend/dev out (engine/shipping.py), so an
  // installed copy has nothing to check here and passes without it, keeping the test counts equal in both trees.
  let html;
  try { html = await readFile(new URL('../dev/qc-demo.html', import.meta.url), 'utf8'); }
  catch (error) { if (error.code === 'ENOENT') return; throw error; }
  const e = browser(t, { reduced: true }), events = new Map();
  window.addEventListener = (name, fn) => events.set(name, fn);
  const nodes = {};
  for (const [, tag, id] of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const node = e.document.createElement(tag); node.id = id; node.clientWidth = 624; node.clientHeight = 560;
    e.document.body.append(node); nodes[id] = node;
  }
  e.document.body.clientWidth = 656; nodes.example.value = 'standard'; nodes.progress.value = '20';
  let demo = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
  demo = demo.replace("'../src/qc-panel.js'", JSON.stringify(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`));
  for (const file of ['qc-chip.mjs', 'qc-copy.mjs', 'qc-chart.mjs']) demo = demo.replace(`'../src/${file}'`, JSON.stringify(new URL(`../src/${file}`, import.meta.url).href));
  await import(`data:text/javascript;base64,${Buffer.from(demo).toString('base64')}`);
  const demoPanel = nodes.panel.childNodes[0]; assert.equal(demoPanel.dataset.state, 'live');
  for (const status of PANEL_STATES) { nodes.state.value = status; nodes.state.emit('change'); assert.equal(demoPanel.dataset.state, status); }
  nodes.state.value = 'live';
  for (const example of ['early','start','nuclei','too-few','low-depth','trial','less','more','no-change','forming-trial','mixed-up','kept','undone','previous','restarting','run-limit','not-counts','software','calm']) {
    nodes.example.value = example; assert.doesNotThrow(() => nodes.example.emit('change'), example);
  }
  assert.equal(e.requested, 0); assert.equal(e.tasks.size, 0);
  events.get('pagehide')();
});

test('an ended run gives the whole tissue\'s numbers only: no line names a flagged area, nor says none stood out', t => {
  const e = browser(t);
  for (const [runState, first] of [['done', 'Finished at iteration 400: 11 in 100 cells looked mixed'], ['stopped', 'Stopped at iteration 400: 11 in 100 cells looked mixed']]) {
    for (const areas of [state().areas, [state().areas[0]], []]) {
      e.panel.setState({ ...state(), state: 'finished', runState, areas });
      const lines = e.panel.finished.childNodes.map(n => n.textContent);
      assert.ok(lines[0].startsWith(first), lines[0]);
      assert.deepEqual(lines.slice(1), [], `${runState}, ${areas.length} areas: that line alone (its number is proseg's own count, as saved)`);
    }
  }
});

test("the chart's tokens are read when the panel and the chip are made, never inside a result's paint", t => {
  let reads = 0;
  const original = Object.getOwnPropertyDescriptor(globalThis, 'getComputedStyle');
  Object.defineProperty(globalThis, 'getComputedStyle', { value: () => { reads++; return { getPropertyValue: () => '' }; }, writable: true, configurable: true });
  t.after(() => { if (original) Object.defineProperty(globalThis, 'getComputedStyle', original); else delete globalThis.getComputedStyle; });
  const e = browser(t);
  const chip = new QcChip(e.document.createElement('div'));
  const made = reads;
  assert.ok(made >= 2, 'read once each, when made');
  e.panel.setState(state()); chip.setState(state()); e.panel.setState(state(12, 410)); chip.setState(state(12, 410));
  assert.equal(reads, made, 'no computed-style read while results are applied');
});

test('while cells form there is no comparison with the start: no change line, no arrow on the chip', t => {
  const e = browser(t), host = e.document.createElement('div'), chip = new QcChip(host);
  const forming = state(7, 40); forming.summary.phase = 'burnin'; forming.areas = []; e.panel.setState(forming); chip.setState(forming);
  assert.equal(e.panel.changeNode.hidden, true, 'not "▼ 3 fewer than at the start"'); assert.ok(e.visible().includes(C.forming));
  assert.equal(chip.label.textContent, 'Mixing 7 in 100 · cells forming');
  assert.match(chip.root.title, /cells are still forming, so compare with the start once they settle/);
  const startView = state(9, 110); startView.areaView = 'start'; e.panel.setState(startView); chip.setState(startView);
  assert.equal(e.panel.changeNode.hidden, true, 'until burn-in + 20 (the start line\'s areas still shown)');
  assert.equal(chip.label.textContent, 'Mixing 9 in 100 · cells forming');
  const formed = state(10); formed.summary.change = { matched: 0.3, threshold: 0.7, meaningful: false }; e.panel.setState(formed); chip.setState(formed);
  assert.equal(e.panel.changeNode.textContent, 'Same as at the start, like for like'); assert.equal(chip.label.textContent, 'Mixing 10 in 100 · same as at the start');
  const formedEarly = state(7, 40); formedEarly.summary.phase = 'burnin'; formedEarly.areas = []; formedEarly.summary.change = { matched: -2, threshold: 0.7, meaningful: true };
  e.panel.setState(formedEarly); assert.equal(e.panel.changeNode.hidden, true, 'the engine\'s change too waits until cells have formed');
  // A run with no start line (from cells with their nuclei, or nuclei alone): the number rises at first, and nothing is
  // said of comparing it with a start it does not have.
  const paired = { ...forming, startKind: 'paired', start: null, summary: { ...forming.summary, start: null } };
  e.panel.setState(paired); chip.setState(paired);
  assert.ok(e.visible().includes(C.formingNoStart)); assert.ok(!e.visible().includes(C.forming)); assert.ok(e.visible().includes(C.nucleiCellsStart));
  assert.equal(chip.label.textContent, 'Mixing 7 in 100 · cells forming');
  assert.equal(chip.root.title, `${copy('chipTooltipFormingNoStart', { n: '7' })} ${copy('chipSmoothed', { i: '40' })} ${C.chipCountAveraged}`);
  assert.doesNotMatch(chip.root.title + C.formingNoStart, /compare with the start/i);
});

test('while cells form the panel says so in words true for each start: "from their nuclei" only for a start from nuclei, alone or with their cells (F4)', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } });
  const forming = (extra = {}) => { const s = state(7, 40, extra); s.summary.phase = 'burnin'; s.areas = []; return { ...s, suggestion: null }; };
  const nuclei = /nuclei|nucleus/i;
  // A start from cells only (the only kind with a start line): no nuclei to form from, in the notice or the slot.
  const cells = forming({ startSource: 'cells', startKind: 'cells' });
  e.panel.setState(cells);
  assert.ok(e.visible().includes(C.forming), e.visible()); assert.ok(e.visible().includes(C.tooEarly));
  assert.doesNotMatch(e.visible(), nuclei, 'nothing about nuclei on a run that has none');
  assert.equal(C.forming, 'Cells are still forming, so this number rises at first. Compare it with the start once they settle.');
  assert.equal(C.tooEarly, 'Too early to suggest anything: cells are still forming.');
  // ... its start line not in yet: the same words, without the start to compare with.
  e.panel.setState({ ...cells, start: null, summary: { ...cells.summary, start: null } });
  assert.ok(e.visible().includes(C.formingShort)); assert.ok(!e.visible().includes(C.forming)); assert.doesNotMatch(e.visible(), nuclei);
  assert.equal(C.formingShort, 'Cells are still forming, so this number rises at first.');
  // Cells with their nuclei, and nuclei alone: cells do form from their nuclei there, said so (nuclei alone also by the
  // engine's start kind alone, a start source the page does not know as nuclei).
  for (const extra of [{ startSource: 'xenium', startKind: 'paired' }, { startSource: 'nuclei', startKind: 'nuclei' }, { startSource: 'nuclei' },
    { startSource: 'xenium', startKind: 'nuclei' }]) {
    const own = forming(extra);
    e.panel.setState({ ...own, start: null, summary: { ...own.summary, start: null } });
    assert.ok(e.visible().includes(C.formingNoStart), JSON.stringify(extra)); assert.ok(e.visible().includes(C.tooEarlyNuclei), JSON.stringify(extra));
    assert.ok(!e.visible().includes(C.tooEarly), 'the slot says it once, in its own words');
  }
  assert.equal(C.formingNoStart, 'Cells are still forming from their nuclei, so this number rises at first.');
  assert.equal(C.tooEarlyNuclei, 'Too early to suggest anything: cells are still forming from their nuclei.');
  // Back to a start from cells only: the slot is written again in its words.
  e.panel.setState(cells); assert.ok(e.visible().includes(C.tooEarly)); assert.doesNotMatch(e.visible(), nuclei);
  for (const key of ['forming', 'formingShort', 'formingNoStart', 'tooEarly', 'tooEarlyNuclei']) assert.deepEqual(bannedWords(C[key]), [], key);
});

test('an ended run: finished, stopped or failed in its own words; one that ended while cells formed is not compared', t => {
  const e = browser(t);
  e.panel.setState({ ...state(), state: 'finished', runState: 'done' });
  // H3: the start is proseg's count at iteration 0, named so.
  assert.ok(e.visible().includes('Finished at iteration 400: 11 in 100 cells looked mixed (10 at iteration 0).'));
  assert.ok(!/cleaner/.test(e.visible()), 'no "saved results are usually a little cleaner": the number is proseg\'s own count');
  e.panel.setState({ ...state(), state: 'finished', runState: 'stopped' });
  assert.ok(e.visible().includes('Stopped at iteration 400: 11 in 100 cells looked mixed (10 at iteration 0).'));
  assert.doesNotMatch(e.visible(), /Flagged|Still flagged|\bareas?\b/i, 'the whole tissue only');
  // A run with no start line (it started from nuclei, alone or with their cells): no start number beside its own.
  for (const none of [{ startKind: 'paired' }, { startSource: 'nuclei', start: null }]) {
    e.panel.setState({ ...state(), ...none, summary: { ...state().summary, start: null }, state: 'finished', runState: 'done' });
    assert.equal(e.panel.finished.textContent, 'Finished at iteration 400: 11 in 100 cells looked mixed.', JSON.stringify(none));
    e.panel.setState({ ...state(), ...none, summary: { ...state().summary, start: null }, state: 'finished', runState: 'stopped' });
    assert.equal(e.panel.finished.textContent, 'Stopped at iteration 400: 11 in 100 cells looked mixed.');
  }
  e.panel.setState({ ...state(), state: 'finished', runState: 'failed', areas: [] });
  assert.ok(e.visible().includes('Stopped at iteration 400')); assert.ok(!e.visible().includes('Finished at'));
  assert.doesNotMatch(e.visible(), /No area stood out/);
  const early = { ...state(11, 40), state: 'finished', runState: 'done', areaView: 'start',
    areas: 'ABCDEFGH'.split('').map(L => ({ ...state().areas[1], L })) };
  early.summary.phase = 'burnin'; e.panel.setState(early);
  assert.ok(e.visible().includes("Finished at iteration 40, while cells were still forming, so it can't be compared with the start yet."));
  assert.doesNotMatch(e.visible(), /already mixed in|A, B, C/i, "the start line's areas are not named");
  assert.ok(!e.visible().includes('Still flagged at the end'), 'the start line\'s areas are not "still flagged"');
  assert.equal(e.panel.changeNode.hidden, true, 'and no "▼ fewer than at the start"');
});

test('a run with no start line (from cells with their nuclei, or nuclei alone): every state honest and quiet, never a comparison it cannot make', t => {
  // The engine's own readout of a drift trial on such a run (its rows raw shares, notes no_matched and small_area: no
  // like-for-like start exists), and its no_history case: the readout says in a few words that no verdict comes on this
  // kind of run, why in the tooltip; no rows headed like for like, no count of iterations for a verdict that never
  // comes, no "too few here, the whole tissue is the better guide", no blame on the iterations before the change.
  const e = browser(t, { options: { suggestions: true } });
  const trial = { id: 1, rule: 'R3', area: 'A', flag: 'no-diffusion', from: true, to: false, asked_seq: 139, start_seq: 150, start_iter: 150,
    rows: [{ kind: 'area', area: 'A', expected: 28, now: 28 }, { kind: 'depth', expected: 300, now: 300 }], verdict: null, on: 'whole',
    notes: ['no_matched', 'small_area'], small_area: true, depth_loss: 8, band: null, results: 47, horizon_iter: 450, final: false, timing: 'later', later_s: 500 };
  const base = state(13, 200), claims = [C.trialSmall, C.trialColumns, C.trialNoHistory, C.trialNoMatched, 'iterations so far', 'like for like',
    'compared with the start', 'than at the start', 'since the start', 'did not move them', 'lost'];
  const claimed = text => claims.filter(c => text.includes(c));
  // The whole panel holds "How is this estimated?" too, whose fourth point says how the check compares in general.
  const panelClaimed = () => claimed(e.visible().replace(C.how4, ''));
  for (const none of [{ startKind: 'paired' }, { startSource: 'nuclei' }]) {
    const run = { ...base, ...none, start: null, summary: { ...base.summary, start: null, retention: null,
      change: { raw: null, matched: null, threshold: 1, model_threshold: 0.6, meaningful: false, direction: null, enough: false } } };
    for (const notes of [['no_matched', 'small_area'], ['no_history'], ['forming'], []]) {
      e.panel.setState({ ...run, trial: { ...trial, notes, small_area: notes.includes('small_area') } });
      const readout = e.visible(e.panel.trial);
      assert.deepEqual(claimed(readout), [], `${JSON.stringify(none)} ${notes}: ${readout}`);
      assert.ok(readout.includes(C.trialNoStart), readout);
      const verdict = e.panel.trial.all('p').find(p => p.textContent === C.trialNoStart);
      assert.equal(verdict.title, C.trialNoStartTooltip, 'why in its tooltip');
      assert.ok(!readout.includes('→') && !readout.includes('Transcripts per cell'), `no rows: ${readout}`);
      assert.ok(e.find('keep') && e.find('undo'), 'Keep and Undo stay');
    }
    // The one card such a run can get (drift, R3: the others need a start line to tag an area) claims nothing of it.
    e.panel.setState({ ...run, trial: null, suggestion: { rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'rough',
      timing: 'later', actions: ['try', 'why'], main: 'try' } });
    assert.ok(e.visible().includes(C.ruleR3Title)); assert.deepEqual(panelClaimed(), [], e.visible());
    // The whole panel says nothing it cannot measure, whatever the trial's notes; the end likewise, also while cells
    // were still forming (its phase or the engine's area_view "start").
    assert.deepEqual(panelClaimed(), [], e.visible());
    const forming = { ...run, trial: null, summary: { ...run.summary, phase: 'burnin', iter: 20 } };
    for (const runState of ['done', 'stopped']) {
      e.panel.setState({ ...forming, state: 'finished', runState });
      assert.equal(e.panel.finished.textContent, `${runState === 'done' ? 'Finished' : 'Stopped'} at iteration 20: 13 in 100 cells looked mixed.`);
      e.panel.setState({ ...run, trial: null, areaView: 'start', state: 'finished', runState });
      assert.deepEqual(panelClaimed(), [], e.visible());
    }
  }
  assert.deepEqual(bannedWords(C.trialNoStart + C.trialNoStartTooltip), []);
  // A run from cells alone keeps its trial readout and its end's comparison as ever.
  e.panel.setState({ ...base, trial });
  assert.ok(e.visible(e.panel.trial).includes(C.trialColumns) && e.visible(e.panel.trial).includes(C.trialNoMatched));
  assert.ok(!e.visible(e.panel.trial).includes(C.trialNoStart));
});

test("the run's check could not be reached: said in plain words with Try again; the switch says neither on nor off meanwhile", t => {
  const e = browser(t);
  e.panel.setState({ state: 'loading' });
  assert.equal(e.panel.toggle.textContent, '', 'not "off" while the engine has not said'); assert.equal(e.panel.toggle.getAttribute('aria-checked'), 'mixed');
  assert.equal(e.panel.toggle.disabled, true);
  e.panel.setState(fromView(null, { failed: true }));
  assert.ok(e.visible().includes(C.loadFailed)); e.find('reload').click(); assert.deepEqual(e.events.at(-1), ['onReload']);
  assert.equal(fromView({ status: 'loading', failed: true }).failed, true);
});

test("a snapshot alone never renders the panel again; the restarting line names the result's iteration", t => {
  const e = browser(t), view = { status: 'live', enabled: true, runId: 'r', model: 'm', summary: state().summary, start: state().start,
    startSource: 'xenium', atStart: false, line: state().series, lineVersion: 4, ticks: [], formingUntil: 100, frameIter: 400, areas: [] };
  e.panel.setState(fromView(view)); const clears = e.clearCount(), status = e.panel.status.childNodes;
  e.panel.setState(fromView({ ...view, frameIter: 401 })); e.panel.setState(fromView({ ...view, frameIter: 402 }));
  assert.equal(e.clearCount(), clears, 'the chart is not drawn again'); assert.equal(e.panel.status.childNodes, status);
  assert.equal('frameIter' in fromView(view), false);
  // Looking back at a result of another frame (`snapped`), the frame's iteration comes along: a scrub changes it anyway.
  assert.deepEqual([fromView({ ...view, snapped: 390 }).snapped, fromView({ ...view, snapped: 390 }).frameIter], [390, 400]);
  e.panel.setState({ ...state(), restarting: true });
  assert.ok(e.visible().includes('Mixing check paused while the graphics worker restarts; showing iteration 400.'));
});

test("once cells have formed nothing is said of areas (none is shown since 2026-10-22); fewer transcripts are said beside the number", t => {
  const e = browser(t);
  e.panel.setState({ ...state(11, 125), areaView: 'smoothed' });
  assert.equal(e.panel.notices.textContent, '', 'not "the outlines now show this run\'s own areas"');
  // The transcripts lost are said beside the number from the engine's retention over all cells, no
  // longer from the typical judged cell; in view when that by itself moves the number, else in the ⓘ.
  const thinner = state(9); thinner.summary.median_tx = 330; thinner.summary.retention = { ratio: 0.85 }; e.panel.setState(thinner);
  assert.ok(e.visible().includes('Cells hold 85 % of their start RNA')); assert.equal(e.panel.retentionInfo.title, `${C.retentionDownTooltip} ${C.retentionBasis}`);
  const same = state(9); same.summary.median_tx = 330; same.summary.retention = { ratio: 0.95 }; e.panel.setState(same);
  assert.ok(!e.visible().includes('Cells hold'), 'under 10 %: not in view');
  assert.ok(e.panel.numberInfo.title.includes('Cells hold 95 % of the transcripts they started with.'), 'just the fact, in the ⓘ');
  const unknown = state(9); unknown.summary.median_tx = 330; e.panel.setState(unknown);
  assert.ok(!e.visible().includes('Cells hold') && !e.panel.numberInfo.title.includes('Cells hold'), 'no retention: nothing claimed');
});

test("calm motion is the page's choice: the button asks, the state it passes back stops the panel's motion", t => {
  const e = browser(t); e.panel.setState({ ...state(8), motionBudgetExceeded: true });
  e.find('calm-motion').click(); assert.deepEqual(e.events.at(-1), ['onCalmMotion', true]);
  assert.equal(e.panel.state.calmMotion, undefined, 'the panel does not keep it by itself (the next render would drop it)');
  e.panel.setState({ ...state(12, 410), motionBudgetExceeded: true, calmMotion: true, motion: true });
  assert.equal(e.panel.numberNode.textContent, '12 in 100'); assert.equal(e.tasks.size, 0, 'nothing moves');
  assert.equal(e.panel.calm.getAttribute('aria-pressed'), 'true');
});

test('a trial row\'s arrow only where the engine\'s verdict says the change is clear, on the row it judged', t => {
  const trial = { id: 3, rule: 'R4', area: 'A', on: 'whole', verdict: null, rows: [] };
  const area = { kind: 'area', area: 'A', expected: 23, now: 22.9 }, whole = { kind: 'whole', expected: 11.2, now: 10.6 }, depth = { kind: 'depth', expected: 340, now: 309 };
  assert.equal(trialRowDirection(trial, whole), '', 'no verdict yet: no arrow for a 0.6 drop');
  assert.equal(trialRowDirection({ ...trial, verdict: 'no_change' }, whole), '');
  assert.equal(trialRowDirection({ ...trial, verdict: 'less' }, whole), 'down'); assert.equal(trialRowDirection({ ...trial, verdict: 'more' }, whole), 'up');
  assert.equal(trialRowDirection({ ...trial, verdict: 'less' }, area), '', 'the verdict is the whole tissue\'s: the area row has none');
  assert.equal(trialRowDirection({ ...trial, verdict: 'less', on: 'area' }, area), 'down'); assert.equal(trialRowDirection({ ...trial, verdict: 'less', on: 'area' }, whole), '');
  assert.equal(trialRowDirection(trial, depth), 'down', 'transcripts per cell: a fact from a 5 % change on');
  assert.equal(trialRowDirection(trial, { kind: 'depth', expected: 340, now: 330 }), '', 'under 5 %');
  assert.equal(trialRowDirection(trial, { kind: 'whole', expected: null, now: 3 }), '');
  const e = browser(t, { options: { suggestions: true, knobLabel } });
  e.panel.setState({ ...state(10.6, 400), trial: { ...trial, flag: 'cell-compactness', from: .04, to: .02, start_iter: 338, rows: [area, whole, depth] } });
  const glyphs = () => e.panel.trial.all().filter(n => n.className === 'qc-row-change').map(n => n.hidden ? '' : n.dataset.direction);
  assert.deepEqual(glyphs(), ['', '', 'down']);
  e.panel.setState({ ...state(10.6, 410), trial: { ...trial, flag: 'cell-compactness', from: .04, to: .02, start_iter: 338, rows: [area, whole, depth], verdict: 'less' } });
  assert.deepEqual(glyphs(), ['', 'down', 'down']);
});

test("a card whose cause is not known points to its cells (the engine's inspect): the page moves the view to its area, at every press", t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  // The engine's cards of this round (engine/qc_rules.py actions_of): R1 shows its starting outlines, then its cells;
  // R6 and R7 their cells, as the main action; their confidence the engine's "unknownCause" / "seen", shown in the card's own words.
  const r1 = { id: 'R1::', rule: 'R1', areas: ['B'], confidence: 'unknownCause', timing: 'later', actions: ['show_start', 'inspect'], main: 'show_start' };
  e.panel.setState({ ...data, suggestion: r1 });
  let card = cardOf(e);
  assert.deepEqual(card.all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), ['show-start', 'inspect', 'why']);
  inOrder(card.textContent, [C.ruleR1Title, C.closerLook, "Show the outlines of Xenium's cells here", C.inspect]);
  const inspect = e.find('inspect');
  assert.equal(inspect.textContent, 'Look at its cells'); assert.equal(inspect.title, C.inspectTooltip);
  assert.match(inspect.title, /^Moves the view to where mixing stands out/, 'where the view goes: no outline is drawn there');
  assert.equal('primary' in inspect.dataset, false, "R1's main action is its starting outlines");
  inspect.click(); assert.deepEqual(e.events.at(-1), ['onInspect', r1, data.areas[1]], 'the card and its area (its outline the result\'s)');
  const before = e.events.length;
  e.find('inspect').click();
  assert.equal(e.events.length, before + 1, 'pressed again: the view moves there again (nothing stays chosen)');
  for (const rule of ['R6', 'R7']) {
    const s = { id: `${rule}::`, rule, areas: ['A'], confidence: rule === 'R6' ? 'unknownCause' : 'seen', timing: 'later', actions: ['inspect'], main: 'inspect',
      ...(rule === 'R7' ? { tried: { flag: 'cell-compactness', from: .04, to: .02 } } : {}) };
    e.panel.setState({ ...data, suggestion: s });
    card = cardOf(e);
    assert.deepEqual(card.all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), ['inspect', 'why'], rule);
    assert.equal('primary' in e.find('inspect').dataset, true, `${rule}: the engine's main action`);
    assert.ok(card.textContent.includes(rule === 'R6' ? C.closerLook : C.seen), rule);
    assert.deepEqual(e.panel.shown.card.actions, ['inspect']); assert.equal(e.panel.shown.card.primary, 'inspect');
  }
  // A card for the whole tissue has no area to move to: no such button.
  assert.deepEqual(suggestionModel({ rule: 'R6', areas: [], actions: ['inspect'], main: 'inspect' }, data).actions, []);
  assert.deepEqual(suggestionModel({ rule: 'R6', areas: ['A'], actions: ['inspect'], main: 'inspect' }, data, { inspect: false }).actions, []);
  // A page that cannot move its view there offers no dead button (the area list's click went with the list).
  const none = browser(t, { options: { suggestions: true, knobLabel, onInspect: undefined } });
  const r6 = { id: 'R6::', rule: 'R6', areas: ['A'], confidence: 'unknownCause', actions: ['inspect'], main: 'inspect' };
  none.panel.setState({ ...data, suggestion: r6 });
  assert.deepEqual(cardOf(none).all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), ['why']);
  none.panel.setState({ ...data, suggestion: r1 });
  assert.deepEqual(cardOf(none).all().filter(n => n.tagName === 'BUTTON').map(n => n.dataset.action), ['show-start', 'why'], 'R1\'s starting outlines stay');
});

test("no change is claimed where the engine gives no verdict (too few cells, cells forming, none yet); a trial without like-for-like numbers says so", t => {
  const e = browser(t), host = e.document.createElement('div'), chip = new QcChip(host);
  const small = state(); small.summary.change = { raw: -2, matched: -2.6, threshold: 2.4, model_threshold: 0.6, meaningful: false, direction: null, enough: false };
  e.panel.setState(small); chip.setState(small);
  assert.equal(e.panel.changeNode.hidden, true); assert.equal(chip.label.textContent, 'Mixing 11 in 100', 'no arrow, and not "about the same" either');
  const notYet = state(); notYet.summary.change = { raw: -1, matched: null, threshold: 0.7, meaningful: false, direction: null, enough: true };
  e.panel.setState(notYet); assert.equal(e.panel.changeNode.hidden, true);
  assert.equal(trialVerdictText({ id: 1, rule: 'R4', notes: ['no_matched'], verdict: null, timing: 'later' }), C.trialNoMatched);
});

test('2026-10-22: no Areas block, row, letter or area sentence in the panel, in any state', t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } });
  // No section, row, tag, card or "small spots" for areas is built at all.
  assert.equal(e.panel.areas, undefined); assert.equal(e.panel.areaRows, undefined);
  const areaNodes = () => e.panel.root.all().filter(n => /qc-area|qc-tag|qc-more-areas|qc-card-area/.test(n.className ?? '')
    || ['area', 'start-outlines'].includes(n.dataset?.action));
  const words = () => e.panel.root.all().flatMap(n => [n.text, n.title, n.getAttribute?.('aria-label')]).filter(Boolean).join('\n');
  // An area named: the word, a letter as the list, the chips, the cards or the trials had it ("in A", "A and C", "A · 23",
  // "Area A"), "No area stands out", "small spots", "flagged" (letters are capitals: "in a test" is not one).
  const named = text => /\bareas?\b|No area|small spots?|flagged/i.test(text) || /\b(?:in|for|of) [A-H]\b(?![-'\w])|\b[A-H] and [A-H]\b|\b[A-H] · \d|\bArea [A-H]\b/.test(text);
  const many = state();
  many.areas = 'ABCDEFGH'.split('').map((L, i) => ({ L, tag: ['rose', 'already', 'similar'][i % 3], edge: i === 0, per100: 30 - i, start_per100: 10,
    judged: 400 - i, growth: .2, tiny: 3, tiny_start: 1, churn: 2, depth: 300, depth_start: 380, outline: outline(i * 400, 0) }));
  const forming = state(4, 40); forming.summary.phase = 'burnin';
  const variants = [state(), many, { ...state(), areas: [] }, { ...state(), areaView: 'single' }, { ...state(), state: 'scrubbing', areaView: 'single' },
    { ...state(), state: 'paused', atStart: true, summary: null },
    { ...forming, areaView: 'start' }, { ...state(), state: 'finished' }, { ...state(), state: 'finished', runState: 'stopped' },
    { ...state(), state: 'finished', areas: [] }, { ...state(11, 125), areaView: 'smoothed' }, { ...state(), view3d: true, overview: true }];
  for (const v of variants) {
    e.panel.setState(v);
    assert.deepEqual(areaNodes(), [], `${v.state}: no area node`);
    assert.equal(named(words()), false, `${v.state}: ${words().slice(0, 300)}`);
  }
  // A card about two areas, and a trial watching one: "where mixing stands out", never a letter.
  e.panel.setState({ ...state(), suggestion: r4('later', { areas: ['A', 'B'], growth: undefined }) });
  assert.ok(cardOf(e).textContent.includes('Cells where mixing stands out grew about 3–18 % bigger'), cardOf(e).textContent);
  assert.equal(named(words()), false, words());
  const trial = { id: 9, rule: 'R4', area: 'A', on: 'area', flag: 'cell-compactness', from: .04, to: .02, start_iter: 338, verdict: 'less', notes: ['small_area'],
    rows: [{ kind: 'area', area: 'A', expected: 23, now: 21.4 }, { kind: 'whole', expected: 11.2, now: 10.6 }], timing: 'later' };
  e.panel.setState({ ...state(10.6, 400), trial });
  inOrder(e.panel.trial.textContent, ['Watching where mixing stands out and the whole tissue', 'Where mixing stands out: 23.0 → 21.4 in 100',
    'Whole tissue: 11.2 → 10.6 in 100', C.trialLessArea, C.trialSmall]);
  assert.equal(named(words()), false, words());
  // Every check above would see a letter or an area where the panel showed them before 2026-10-22.
  for (const old of ['Areas (2)', 'A · 23 in 100 (start 14) · 310 cells', 'for area A', 'cells in A have grown', 'Watching area A and the whole tissue',
    'Area A: 23.0 → 21.4 in 100', 'Still flagged at the end: areas A and B.', '+2 small spots', 'No area stands out from the rest of the tissue.'])
    assert.equal(named(old), true, old);
});

// ---- 2026-10-31: the count proseg saves (decision A), smoothed and own numbers (H2, H10), outline-era figures (H5) ----
test("a saved iteration's numbers say they are the count proseg saves; its arrow and retention that they read the averaged count, like for like; the same after Resume", t => {
  // Decision A: an iteration counted for a pause, the hold, a finish or the end (count "sampled") shows the count proseg
  // saves there (the engine's single result); its change and retention compare the averaged count like for like.
  const e = browser(t), data = state();
  const start = `At the start (proseg's count at iteration 0): 10 in 100. ${C.startTooltip}`;
  data.summary = { ...data.summary, count: 'sampled', single: true, retention: { ratio: 0.92 },
    change: { matched: -1.84, threshold: 0.7, model_threshold: 0.6, meaningful: true, direction: 'fewer' } };
  e.panel.setState({ ...data, state: 'paused' });
  const said = () => [e.panel.numberInfo.title, e.panel.changeNode.title, e.panel.changeNode.textContent];
  assert.deepEqual(e.panel.numberInfo.title.split('\n'), [copy('numberOwn', { i: '400' }), C.numberTooltip, C.meaning,
    "Based on proseg's count (as saved): 3,188 cells with 100 or more transcripts · typical cell 340 (386 at the start).", C.basisTooltipSaved,
    `Cells hold 92 % of the transcripts they started with. ${C.retentionBasis} ${C.retentionSaved}`, "Proseg's cells saved in this sample: 12 in 100."]);
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${C.likeForLikeSaved}`, 'its own change: never smoothed, never held');
  assert.equal(e.panel.changeNode.textContent, '▼ 1.8 fewer, like for like');
  const paused = said();
  // Scrubbed back to it after Resume (the run running): the same words; none depends on the run's state.
  e.panel.setState({ ...data, state: 'scrubbing', runState: 'running', frameIter: 900 });
  assert.deepEqual(said(), paused);
  // The retention line in view says it too, in its ⓘ.
  e.panel.setState({ ...data, summary: { ...data.summary, retention: { ratio: 0.72 } } });
  assert.equal(e.panel.retentionInfo.title, `${C.retentionDownTooltip} ${C.retentionBasis} ${C.retentionSaved}`);
  // The averaged count's own words, elsewhere.
  e.panel.setState({ ...state(), summary: { ...state().summary, retention: { ratio: 0.72 }, change: data.summary.change } });
  assert.ok(e.panel.numberInfo.title.includes("\nBased on proseg's count (averaged): ") && e.panel.numberInfo.title.includes(`\n${C.basisTooltip}\n`));
  assert.ok(![C.basisTooltipSaved, C.likeForLikeSaved, C.retentionSaved].some(w => [...said(), e.panel.retentionInfo.title].join('\n').includes(w)));
});

test("the verdict line says what it compares with the change's own two numbers, what they are in its ⓘ; a summary without them keeps its words (P2)", t => {
  // At a saved iteration the headline reads the count proseg saves (11 in 100) and the start line iteration 0's (10 in
  // 100), while the change compares the averaged count like for like: its own two numbers (per100_now, per100_start).
  const e = browser(t), data = state();
  const start = `At the start (proseg's count at iteration 0): 10 in 100. ${C.startTooltip}`;
  const change = { matched: -1.8, threshold: 0.7, model_threshold: 0.6, meaningful: true, direction: 'fewer', per100_now: 6.8125, per100_start: 8.6125 };
  e.panel.setState({ ...data, state: 'paused', summary: { ...data.summary, count: 'sampled', single: true, change } });
  assert.equal(e.panel.changeNode.textContent, '▼ Fewer, like for like: 6.8 now, 8.6 at the start');
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${C.likeForLikeNumbers}\n${C.likeForLikeSaved}`);
  // Live: smoothed (the summary's own numbers move with it); about the same says its numbers too.
  e.panel.setState(state(11, 400, { summary: { ...data.summary, change: { ...change, matched: -0.3, meaningful: false, direction: 'same', per100_now: 8.31, per100_start: 8.6125 } } }));
  assert.equal(e.panel.changeNode.textContent, 'About the same, like for like: 8.3 now, 8.6 at the start');
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${C.likeForLikeNumbers}\n${C.changeSmoothed}`);
  // A summary stored before them: the line and its ⓘ as before.
  e.panel.setState({ ...data, state: 'paused', summary: { ...data.summary, count: 'sampled', single: true, change: { ...change, per100_now: undefined, per100_start: undefined } } });
  assert.equal(e.panel.changeNode.textContent, '▼ 1.8 fewer, like for like');
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${C.likeForLikeSaved}`);
  // Live, its word held a few seconds (the engine's Steady: the change itself reads more now): the line's words without
  // numbers, whatever numbers the block carries (here 9.5 against 8.6; the engine's are those the word compared, an
  // earlier result's); its ⓘ says the word is held and what the change reads, and nothing of two numbers not shown.
  const heldChange = { ...change, matched: 0.9, held: true, direction_now: 'more', meaningful_now: true, held_s: 3.2, per100_now: 9.5, per100_start: 8.6 };
  e.panel.setState(state(11, 400, { summary: { ...data.summary, change: heldChange } }));
  assert.equal(e.panel.changeNode.textContent, '▼ Fewer, like for like');
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\n${copy('changeHeld', { now: C.changeNowMore })}\n${C.changeSmoothed}`);
  // ... and with its numbers again once the word is its own.
  e.panel.setState(state(11, 400, { summary: { ...data.summary, change: { ...change, per100_now: 6.9 } } }));
  assert.equal(e.panel.changeNode.textContent, '▼ Fewer, like for like: 6.9 now, 8.6 at the start');
});

test("the headline's ⓘ first names the iteration its number is from and how: live smoothed (with its 0.3 margin), looked back at as shown live, or its own (H10, H2)", t => {
  const e = browser(t), first = () => e.panel.numberInfo.title.split('\n')[0];
  const live = state(11, 400, { summary: { ...state().summary, change: { matched: -1.84, threshold: 0.7, model_threshold: 0.6, meaningful: true, direction: 'fewer' } } });
  e.panel.setState(live);
  assert.equal(first(), copy('numberLive', { i: '400' }));
  assert.match(first(), /^From iteration 400, the newest proseg has counted, smoothed with the iterations before it; its whole number changes only once it is 0\.3 past the halfway point/);
  assert.ok(e.panel.changeNode.title.includes(`\n${C.changeSmoothed}`), 'the live change: smoothed like the number');
  e.panel.setState({ ...live, state: 'scrubbing', frameIter: 420 });
  assert.equal(first(), copy('numberPast', { i: '400' }), 'scrubbed back to a smoothed result: as it was shown live');
  assert.ok(e.panel.changeNode.title.includes(`\n${C.changeSmoothed}`));
  // An own number: a resting run's newest (qc-client.mjs restSeq) or a result judged alone; paused or looked back at.
  for (const status of ['paused', 'scrubbing', 'finished', 'live']) {
    e.panel.setState({ ...live, state: status, summary: { ...live.summary, single: true } });
    assert.equal(first(), copy('numberOwn', { i: '400' }), status); assert.ok(!e.panel.changeNode.title.includes(C.changeSmoothed), status);
  }
  e.panel.setState({ ...live, state: 'paused', atStart: true, summary: null });
  assert.equal(first(), copy('numberOwn', { i: '0' }), 'the start line: iteration 0\'s own');
  // A result that says neither (an older engine's): no first line.
  e.panel.setState({ ...live, summary: { ...live.summary, single: undefined } });
  assert.equal(first(), C.numberTooltip);
  for (const key of ['numberLive', 'numberPast', 'numberOwn']) assert.deepEqual(bannedWords(copy(key, { i: '400' })), [], key);
});

test("a word held a few seconds says so in the arrow's ⓘ, after the rule, with what the change itself reads now; unheld or own, nothing (H2)", t => {
  const e = browser(t), data = state();
  const start = `At the start (proseg's count at iteration 0): 10 in 100. ${C.startTooltip}`;
  data.summary.change = { matched: -0.97, threshold: 1.0, model_threshold: 0.6, meaningful: true, direction: 'fewer', direction_now: 'same', held: true, held_s: 2.1 };
  e.panel.setState(data);
  assert.equal(e.panel.changeNode.textContent, '▼ Fewer, like for like', 'the word held, without a number it need not fit (its size now)');
  assert.equal(e.panel.changeNode.title, `${start}\n${C.changeTooltip}\nThis word is held a few seconds so it doesn't flicker; the change itself now reads about the same as at the start.\n${C.changeSmoothed}`);
  e.panel.setState({ ...data, summary: { ...data.summary, change: { ...data.summary.change, direction: 'same', direction_now: 'more' } } });
  assert.ok(e.panel.changeNode.title.includes(`\n${copy('changeHeld', { now: C.changeNowMore })}\n`));
  for (const change of [{ ...data.summary.change, held: false }, { matched: -0.97, threshold: 1.0, meaningful: true, direction: 'fewer' }]) {
    e.panel.setState({ ...data, summary: { ...data.summary, change } }); assert.ok(!e.panel.changeNode.title.includes('held a few seconds'), JSON.stringify(change));
  }
});

test("figures from the tests made while the check counted the outlines say so in their tooltips (H5): R2, R4 and R5's tests, the sized line, every shape cost and timing line; none of R3's", t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  const tipOf = (card, text) => card.all().find(n => n.text === text)?.title ?? null, era = C.outlineEraTooltip;
  const r2 = { rule: 'R2', areas: ['A'], flag: 'cell-compactness', from: 0.08, to: 0.04, v: 0.08, confidence: 'likely', timing: 'later', actions: ['new_run', 'try'], main: 'new_run' };
  e.panel.setState({ ...data, suggestion: r2 });
  let card = cardOf(e);
  assert.deepEqual([tipOf(card, C.likelySize), tipOf(card, C.ruleR2Why), tipOf(card, C.shapeCost), tipOf(card, C.timingLaterNoTime), tipOf(card, C.suggestionFooterNewRun)],
    [era, era, era, era, ''], 'R2: its sized line, its tests, its cost and its timing; not its footer');
  e.panel.setState({ ...data, suggestion: r4('early') }); card = cardOf(e);
  assert.deepEqual([tipOf(card, C.rough), tipOf(card, C.ruleR4Why), tipOf(card, C.shapeCost), tipOf(card, C.timingEarly)], ['', era, era, era],
    'R4: a rough guess has no figure');
  e.panel.setState({ ...data, suggestion: r4('later', { later_s: 162 }) }); card = cardOf(e);
  assert.equal(tipOf(card, copy('timingLater', { t: '3 min' })), era);
  const r5 = { rule: 'R5', areas: ['A'], flag: 'prior-weight', from: 1, to: 2, k: 4, confidence: 'rough', timing: 'later', actions: ['new_run', 'try'], main: 'new_run' };
  e.panel.setState({ ...data, suggestion: r5 }); card = cardOf(e);
  assert.deepEqual([tipOf(card, C.ruleR5Why), tipOf(card, C.timingUntested)], [era, ''], 'R5: its tests; its untested timing has no figure');
  const r3 = { rule: 'R3', areas: [], flag: 'no-diffusion', from: true, to: false, confidence: 'likely', timing: 'later', actions: ['try'], main: 'try' };
  e.panel.setState({ ...data, suggestion: r3 }); card = cardOf(e);
  assert.deepEqual([tipOf(card, C.likelySize), tipOf(card, C.ruleR3Why), tipOf(card, C.ruleR3More)], ['', '', ''], 'R3: measured on proseg\'s count');
  // The trial's lines with the timing figures say so too; a verdict or a note's line does not.
  const trial = { id: 5, rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02, start_iter: 300, rows: [], verdict: null, on: 'area', notes: [], timing: 'later' };
  const lineTip = text => e.panel.trial.all('p').find(p => p.textContent === text)?.title;
  e.panel.setState({ ...data, trial });
  assert.equal(lineTip(C.trialPendingLaterNoTime), era);
  e.panel.setState({ ...data, trial: { ...trial, timing: 'early' } });
  assert.equal(lineTip(copy('trialNoChange', { timing: C.timingLaterNoTime })), era);
  e.panel.setState({ ...data, trial: { ...trial, verdict: 'no_change', iterations: 40, timing: 'early' } });
  assert.equal(lineTip(copy('trialNoChangeAfter', { k: '40' })), '', 'no figure in it');
  e.panel.setState({ ...data, trial: { ...trial, verdict: 'less' } });
  assert.equal(lineTip(C.trialLessArea), '');
});

test("a card on a saved iteration (count \"sampled\") ends its Why? with what its numbers read: the averaged count, smoothed (cardAveraged); an averaged one's not", t => {
  const e = browser(t, { options: { suggestions: true, knobLabel } }), data = state();
  e.panel.setState({ ...data, suggestion: r4('later') });
  e.find('why').click();
  assert.ok(!e.visible(cardOf(e)).includes(C.cardAveraged));
  e.panel.setState({ ...data, summary: { ...data.summary, count: 'sampled', single: true }, suggestion: r4('later') });
  const card = cardOf(e), lines = card.all().filter(n => n.tagName === 'P' && n.parentElement?.className === 'qc-evidence');
  assert.equal(lines.at(-1).textContent, C.cardAveraged, 'the last paragraph under Why?');
  assert.ok(e.visible(card).includes(C.cardAveraged), 'Why? stays open on the same card');
  assert.equal(suggestionModel(r4('later'), { ...data, summary: { ...data.summary, count: 'sampled' } }).more.at(-1).kind, 'count');
  assert.ok(!suggestionModel(r4('later'), data).more.some(p => p.kind === 'count'));
});

test("a stopped run shows its newest result's own numbers: the stopped line names that result's iteration and number (H2)", t => {
  // qc-client.mjs: proseg stopped on an uncounted frame; the newest result held (iteration 180) shows its own numbers.
  const e = browser(t), data = state(8.79, 180);
  e.panel.setState({ ...data, state: 'finished', runState: 'stopped', summary: { ...data.summary, shown: 9, single: true } });
  assert.equal(e.panel.finished.textContent, 'Stopped at iteration 180: 9 in 100 cells looked mixed (10 at iteration 0).');
  assert.equal(e.panel.numberNode.textContent, '9 in 100');
  assert.equal(e.panel.numberInfo.title.split('\n')[0], copy('numberOwn', { i: '180' }));
});
