import test from 'node:test';
import assert from 'node:assert/strict';
import { QcClient, panelStatus, floorIndex, floorPoint, insertSorted, decimate, formingUntil, changeTicks, cardKey,
  COALESCE_MS, WITH_SNAPSHOT_MS, REQUEST_MS, KEEP_SUMMARIES, MOTION_GAP_MS, MOVE_WINDOW_MS, RETRY_MS, TRIM_POINTS, KEEP_POINTS, ENDED_NOTE_ITERS, READOUT_MS } from '../src/qc-client.mjs';
import { cardLine } from '../src/qc-copy.mjs';

// The mixing check's data on the page (qc-client.mjs): which result is shown for the frame on screen and when.
const settle = async (n = 5) => { for (let i = 0; i < n; i++) await new Promise(resolve => setImmediate(resolve)); };

test('sorted lookups, decimation, the forming stretch and plain-named knob ticks are incremental and exact', () => {
  const seqs = [];
  for (const v of [5, 1, 9, 5, 3]) insertSorted(seqs, v);
  assert.deepEqual(seqs, [1, 3, 5, 9]);
  assert.deepEqual([0, 1, 4, 5, 8, 9, 99].map(t => floorIndex(seqs, t)), [-1, 0, 1, 2, 2, 3, 3]);
  assert.deepEqual([0, 4, 9].map(t => floorPoint(seqs.map(seq => ({ seq })), t)), [-1, 1, 3]);
  const many = Array.from({ length: 1000 }, (_, i) => i), d = decimate(many, 10);
  assert.equal(d.length, 10); assert.equal(d[0], 0); assert.equal(d.at(-1), 999); assert.equal(decimate([1, 2], 10).length, 2);
  const frames = [{ seq: 0, iter: 0, phase: 'init' }, { seq: 1, iter: 1, phase: 'burnin' }];
  let f = formingUntil(null, frames);
  assert.equal(f.until, 1, 'still forming: up to the newest frame');
  frames.push({ seq: 2, iter: 2, phase: 'burnin' }); f = formingUntil(f, frames); assert.equal(f.until, 2);
  frames.push({ seq: 3, iter: 5, phase: 'upscale' }, { seq: 4, iter: 6, phase: 'sampling' }); f = formingUntil(f, frames);
  assert.equal(f.until, 5, 'fixed at the first frame past burn-in'); assert.equal(formingUntil(f, frames).until, 5);
  const live = (c, p) => ({ 'cell-compactness': c, 'prior-weight': p, 'no-diffusion': false, temperature: 1 });
  const list = [{ seq: 0, iter: 0, live: live(0.03999999910593033, 1) }, { seq: 1, iter: 1, live: live(0.03999999910593033, 1) }];
  let ticks = changeTicks(null, list);
  assert.deepEqual(ticks.ticks, []);
  list.push({ seq: 2, iter: 7, live: live(0.019999999552965164, 1) }, { seq: 3, iter: 9, live: live(0.019999999552965164, 4) });
  ticks = changeTicks(ticks, list);
  assert.deepEqual(ticks.ticks.map(t => [t.iter, t.key]), [[7, 'tighterShapes'], [9, 'holdNuclei']]);
  list.push({ seq: 4, iter: 12, live: live(0.03999999910593033, 4) });
  assert.deepEqual(changeTicks(ticks, list).ticks.map(t => t.key), ['tighterShapes', 'holdNuclei', 'standardShapes']);
  assert.notEqual(changeTicks(ticks, [...list]), ticks, 'another frame list starts over');
});

test("the panel's state: the run's check, its sample's learning, the run and the frame on screen", () => {
  const base = { describe: { state: 'ready', sample_state: 'ready' }, sample: null, enabled: true, results: true, start: { per100: 10 },
    run: { state: 'running', ack: { state: 'running' } }, seq: 5, newest: 5, live: true };
  const s = over => panelStatus({ ...base, ...over });
  assert.equal(s({ describe: null }), 'loading');
  assert.equal(s({ enabled: false }), 'off');
  assert.equal(s({ describe: { state: 'off', sample_state: 'off' } }), 'waiting', 'switched on: its sample is queued with it');
  assert.equal(s({ describe: { state: 'waiting', sample_state: 'waiting' }, results: false }), 'waiting');
  assert.equal(s({ describe: { state: 'waiting' }, sample: { state: 'learning' } }), 'learning', 'the sample\'s newer event');
  assert.equal(s({ describe: { state: 'learning' }, sample: { state: 'ready' }, results: false, start: null }), 'waiting-start');
  assert.equal(s({ describe: { state: 'ready' }, sample: { state: 'learning' } }), 'learning', 'learning again (relearn)');
  assert.equal(s({ describe: { state: 'unavailable', sample_state: 'ready' }, sample: { state: 'ready' } }), 'unavailable', 'no transcripts: stays so');
  assert.equal(s({ describe: { state: 'failed' } }), 'failed');
  assert.equal(s({ describe: { state: 'starting' }, results: false, start: null }), 'waiting-start');
  assert.equal(s({ seq: 3, live: false }), 'scrubbing');
  assert.equal(s({ seq: 3, live: true }), 'live', 'following live, the newest frame still downloading: never "scrubbing"');
  assert.equal(s({ run: { state: 'done' } }), 'finished');
  assert.equal(s({ run: { state: 'paused', ack: { state: 'paused' } } }), 'paused');
  assert.equal(s({ run: { state: 'paused', ack: { state: 'running' } } }), 'live', 'pausing: not yet');
});

/** A client against a fake engine, page and clock. */
function rig({ describe = {}, cells = 3000, live = true, wantCells = false } = {}) {
  const calls = [], timers = new Map(), rides = [], marks = [], views = [], notices = [], counts = {};
  let clock = 0, nextId = 1;
  const page = { run: { id: 'r1', sample: 's1', state: 'running', ack: { state: 'running' }, info: { cells } }, frames: [], seq: -1, live, wantCells };
  for (let i = 0; i < 40; i++) page.frames.push({ seq: i, iter: i * 10, phase: i < 5 ? 'burnin' : 'sampling' });
  const answer = { enabled: true, state: 'ready', reason: null, sample_state: 'ready', progress: null, eta_s: null, relearn: false,
    model: { id: 'm1' }, start: { per100: 10, judged: 300, median_tx: 380 }, start_source: 'xenium', batch_per100: 12, latest_seq: null, seqs: [], ...describe };
  const api = {
    runQc: async () => { calls.push('describe'); return answer; },
    qcSeries: async () => { calls.push('series'); return { model: 'm1', points: answer.seqs.map(seq => ({ seq, iter: seq * 10, per100: 9, shown: 9, judged: 300 })) }; },
    setRunQc: async (id, body) => { calls.push(['post', body]); if (body.enabled === false && rigState.refuse) throw new Error('refused'); return { ...answer, ...body }; },
    qcSummary: async (id, seq) => { calls.push(['summary', seq]); if (!answer.seqs.includes(seq)) throw Object.assign(new Error('Not checked yet'), { status: 404 });
      return summary(seq); },
    qcCells: async (id, seq, model, signal, count) => { calls.push(['cells', seq, model, ...(count ? [count] : [])]); return Uint8Array.of(seq === 'start' ? 3 : 80, 255, 20); },
    learnSampleQc: async () => { calls.push('learn'); return { state: 'waiting', reason: 'queue' }; },
    sampleQc: async () => { calls.push('sample'); return { state: 'ready' }; },
    qcStart: async () => { calls.push('start'); if (!rigState.startAnswer) throw Object.assign(new Error('No start line'), { status: 404 }); return rigState.startAnswer; },
    // The trial and previous-run endpoints (engine/qc.py trial, trial_end, previous).
    qcPrevious: async (id, points) => { calls.push(['previous', points]); if (rigState.previousFails) throw Object.assign(new Error('busy'), { status: 503 }); return rigState.previous; },
    qcTrial: async (id, body) => { calls.push(['trial', body]); if (rigState.trialRefused) throw Object.assign(new Error('A trial is already running'), { status: 409 });
      return { id: ++rigState.trials, ...body, asked_seq: 20, asked_iter: 200, start_seq: null, start_iter: null, rows: [], verdict: null, notes: [] }; },
    qcTrialEnd: async (id, outcome, rule) => { calls.push(['end', outcome, rule ?? null]); if (rigState.endStatus) throw Object.assign(new Error('No trial'), { status: rigState.endStatus });
      return { id: rigState.trials, outcome }; },
  };
  const rigState = { refuse: false, startAnswer: null, previous: { run: null, series: [] }, trials: 0, trialRefused: false, endStatus: null, previousFails: false };
  const client = new QcClient({ api, context: () => page, render: v => views.push(v), notice: t => notices.push(t),
    harness: { count: (name, n = 1) => { counts[name] = (counts[name] || 0) + n; } },
    batch: { ride: (key, fn) => rides.push([key, fn]), mark: (key, fn) => marks.push([key, fn]) },
    clock: { now: () => clock, setTimeout: (fn, ms) => { const id = nextId++; timers.set(id, { fn, at: clock + ms }); return id; }, clearTimeout: id => timers.delete(id) } });
  const advance = ms => {
    clock += ms;
    for (;;) { const due = [...timers].filter(([, t]) => t.at <= clock).sort((a, b) => a[1].at - b[1].at)[0]; if (!due) break; timers.delete(due[0]); due[1].fn(); }
  };
  const flush = () => { const jobs = [...rides, ...marks]; rides.length = marks.length = 0; for (const [, fn] of jobs) fn(); };
  return { client, page, calls, timers, rides, marks, views, notices, counts, advance, flush, answer, rigState, view: () => views.at(-1) };
}
const summary = (seq, extra = {}) => ({ seq, iter: seq * 10, phase: 'sampling', model: 'm1', single: false, judged: 300, mixed: 33, per100: 11, shown: 11,
  median_tx: 340, start: { per100: 10 }, areas: [{ L: 'A', tag: 'rose', per100: 20, outline: [[0, 0], [1, 0], [1, 1]] }], more_spots: 0, ...extra });
const ready = (seq, extra) => ({ ev: 'qc', run: 'r1', state: 'ready', seq, iter: seq * 10, summary: summary(seq, extra) });

test("switched off, the view carries the sample's own reason, for the panel to say why it cannot be learned", async () => {
  // The run's answer names the sample's reason with the switch off too (sample_reason); the
  // sample's own events, when they come, say it instead.
  const r = rig({ describe: { enabled: false, state: 'off', reason: null, sample_state: 'unavailable', sample_reason: 'positions' } });
  r.client.reset(r.page.run); await settle();
  assert.equal(r.view().status, 'off');
  assert.equal(r.view().sampleState, 'unavailable'); assert.equal(r.view().sampleReason, 'positions');
  r.client.event({ sample: 's1', state: 'failed', reason: 'too_few_cells' });
  assert.equal(r.view().sampleState, 'failed'); assert.equal(r.view().sampleReason, 'too_few_cells');
});

test("a switch flipped elsewhere (another window, the API) shows at once; this page's own switch decides while it waits", async () => {
  const r = rig({ describe: { seqs: [3], latest_seq: 3 } });
  r.page.seq = 5; r.client.reset(r.page.run); await settle();
  assert.equal(r.view().enabled, true); assert.equal(r.view().summary.seq, 3);
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false });
  assert.equal(r.view().status, 'off'); assert.equal(r.client.applied, null); assert.deepEqual(r.view().areas, []);
  const asked = r.calls.length, renders = r.views.length;
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false });     // what it shows already
  r.client.event({ ev: 'qc', run: 'r2', state: 'switch', enabled: true });      // another run's
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch' });                     // no value
  assert.equal(r.views.length, renders, 'nothing to do'); assert.equal(r.calls.length, asked);
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: true });
  assert.equal(r.view().enabled, true); await settle();
  assert.ok(r.calls.slice(asked).includes('describe'), 'switched on: its answer and its results are asked for again');
  assert.equal(r.view().status, 'live'); assert.equal(r.view().summary.seq, 3);
});

test("the switch's count decides between this page's answer and a push, whatever order they arrive in", async () => {
  const r = rig({ describe: { seqs: [3], latest_seq: 3, switch_n: 4 } });
  r.page.seq = 5; r.client.reset(r.page.run); await settle();
  let release; const gate = () => new Promise(resolve => { release = resolve; });
  // The echo of this page's own change (pushed with the same count as its answer): nothing more happens.
  r.client.api.setRunQc = async (id, body) => { await gate(); return { ...r.answer, enabled: body.enabled, switch_n: 5 }; };
  let post = r.client.setEnabled(false);
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false, n: 5 });
  assert.equal(r.view().status, 'off', 'the click shows at once'); await settle(); release(); await post; await settle();
  assert.equal(r.view().status, 'off');
  // Another window switches it back on while this page's request is on its way: its push (a newer count) wins.
  r.client.api.setRunQc = async (id, body) => { await gate(); return { ...r.answer, enabled: body.enabled, switch_n: 6 }; };
  r.client.applySwitch({ enabled: true, n: 5 }); await settle();
  post = r.client.setEnabled(false);
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: true, n: 7 });
  assert.equal(r.view().status, 'off', 'never undone while the click waits'); await settle(); release(); await post; await settle();
  assert.equal(r.view().enabled, true, 'the engine is on: so is the page'); assert.equal(r.client.switchN, 7);
  // A push older than what the page has (it was overtaken by the answer) changes nothing.
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false, n: 6 });
  assert.equal(r.view().enabled, true);
  // A push heard before the page's first answer: kept, and it wins when it is newer than that answer.
  const late = rig({ describe: { seqs: [3], latest_seq: 3, switch_n: 2 } });
  late.page.seq = 5; late.client.reset(late.page.run);
  late.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false, n: 3 });
  await settle(); assert.equal(late.view().status, 'off'); assert.equal(late.client.switchN, 3);
  const older = rig({ describe: { seqs: [3], latest_seq: 3, switch_n: 3 } });
  older.page.seq = 5; older.client.reset(older.page.run);
  older.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false, n: 2 });
  await settle(); assert.equal(older.view().enabled, true, 'an older push loses to the answer');
});

test('onSwitch: what new runs start with follows this page\'s own switch (once the engine took it) and a flip pushed from another window, never a first look', async () => {
  const r = rig({ describe: { seqs: [3], latest_seq: 3, switch_n: 4 } }), flips = [];
  r.client.onSwitch = on => flips.push(on);
  r.page.seq = 5; r.client.reset(r.page.run); await settle();
  assert.deepEqual(flips, [], 'opening a run is no flip: its switch may be an old one');
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: true, n: 4 });   // the first look after the watch
  assert.deepEqual(flips, []);
  await r.client.setEnabled(false); await settle(); assert.deepEqual(flips, [false]);
  await r.client.setEnabled(true); await settle(); assert.deepEqual(flips, [false, true]);
  // Refused: the engine remembers nothing new.
  r.rigState.refuse = true; await r.client.setEnabled(false); await settle();
  assert.deepEqual(flips, [false, true]); assert.equal(r.view().enabled, true); assert.equal(r.notices.length, 1);
  // Another window flips it (a newer count): new runs follow that too; an older push or the same value changes nothing.
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false, n: 9 }); assert.deepEqual(flips, [false, true, false]);
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: true, n: 8 }); assert.deepEqual(flips, [false, true, false]);
  r.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false, n: 10 }); assert.deepEqual(flips, [false, true, false]);
  // A flip heard before the page's first answer counts once that answer shows it is newer; the constructor takes it too.
  const seen = [], late = rig({ describe: { seqs: [3], latest_seq: 3, switch_n: 2 } });
  late.client.onSwitch = on => seen.push(on);
  late.page.seq = 5; late.client.reset(late.page.run);
  late.client.event({ ev: 'qc', run: 'r1', state: 'switch', enabled: false, n: 3 }); await settle();
  assert.deepEqual(seen, [false]);
  assert.equal(new QcClient({ api: {}, context: () => ({}), onSwitch: Math.sign }).onSwitch, Math.sign);
});

test('opening a run asks for its check, then its line; until a result arrives it waits for one', async () => {
  const r = rig({ describe: { seqs: [3, 6], latest_seq: 6 } });
  r.page.seq = 10; r.client.reset(r.page.run); await settle();
  assert.equal(r.calls[0], 'describe'); assert.ok(r.calls.includes('series'));
  const v = r.view();
  assert.equal(v.status, 'live'); assert.equal(v.summary.seq, 6, 'the newest result at or before frame 10'); assert.equal(v.summary.shown, 9);
  assert.deepEqual(v.line.map(p => p.seq), [3, 6]); assert.equal(v.start.per100, 10); assert.equal(v.formingUntil, 50);
  assert.equal(r.counts.qcApplied, 1);
  // Nothing on the page shows a start number from outside the app, so the sample's answer that
  // carried one is never asked for, and the view has no such number.
  assert.equal(r.calls.filter(c => c === 'sample').length, 0); assert.equal('notebook' in r.view(), false);
  const empty = rig(); empty.page.seq = 2; empty.client.reset(empty.page.run); await settle();
  assert.deepEqual(empty.calls, ['describe'], 'no series to ask for yet (the previous run\'s line, once)');
  assert.equal(empty.view().status, 'live'); assert.equal(empty.view().summary, null);
});

test('a result is never shown on an earlier frame (but looking back before every result, the oldest after it); following live it is applied with the next snapshot', async () => {
  const r = rig(); r.page.seq = 9; r.client.reset(r.page.run); await settle(); r.flush();
  r.client.event(ready(10));
  assert.equal(r.rides.length, 1, 'it rides the next snapshot\'s paint batch'); assert.equal(r.timers.size, 1, '... or WITH_SNAPSHOT_MS later');
  r.flush(); assert.equal(r.view().summary, null, 'frame 9 on screen: result 10 waits'); assert.equal(r.client.applied, null);
  r.page.seq = 10; r.client.frame();                 // main.js: the snapshot of frame 10, in its paint batch
  assert.equal(r.view().summary.seq, 10); assert.equal(r.timers.size, 0, 'the fallback timer is gone'); assert.equal(r.counts.qcApplied, 1);
  r.client.frame(); r.client.frame(); assert.equal(r.counts.qcApplied, 1, 'counted once per result');
  // A run that stops sending snapshots: the result shows anyway, WITH_SNAPSHOT_MS later, in a batch of its own.
  r.client.event(ready(11)); r.page.seq = 11; r.rides.length = 0;
  r.advance(WITH_SNAPSHOT_MS - 1); assert.equal(r.marks.length, 0); r.advance(1); assert.equal(r.marks.length, 1);
  r.flush(); assert.equal(r.view().summary.seq, 11);
  // Back to an earlier frame before every result: the oldest after it (as the engine answers a frame asked for: the
  // nearest with proseg's count), said to be another iteration's (`snapped`); the frame itself asked for once.
  r.page.live = false; r.page.seq = 8; r.client.frame();
  assert.equal(r.view().summary.seq, 10); assert.equal(r.view().snapped, 100); assert.equal(r.view().later, true); assert.equal(r.view().status, 'scrubbing');
  r.client.frame(); r.client.frame(); r.advance(REQUEST_MS);
  assert.deepEqual(r.calls.filter(c => c[0] === 'post'), [['post', { seq: 8 }]], 'asked once (debounced), not on every render');
  r.client.event(ready(8)); assert.equal(r.view().summary.seq, 8, 'not following live: at once'); assert.equal(r.view().snapped, null, 'its own');
  assert.equal(r.view().later, false);
  r.client.event(ready(9, { per100: 30 })); assert.equal(r.view().summary.seq, 8, 'a later result never colours frame 8');
  // An earlier result than the frame on screen, looking back: that one, said so; following live, nothing said (the
  // result is always a little behind the newest frame).
  r.page.seq = 9; r.client.frame(); assert.equal(r.view().summary.seq, 9);
  r.page.seq = 12; r.client.frame(); assert.equal(r.view().summary.seq, 11); assert.equal(r.view().snapped, 110); assert.equal(r.view().later, false, 'an earlier one');
  r.page.seq = 13; r.client.frame(); assert.equal(r.view().snapped, 110); assert.equal(r.view().frameIter, 130);
  r.page.live = true; r.client.frame(); assert.equal(r.view().snapped, null);
});

test('a frame checked but not loaded here shows the nearest earlier result the page has, then its own', async () => {
  const r = rig({ describe: { seqs: [2, 4, 6, 8], latest_seq: 8 }, live: false }); r.page.seq = 8; r.client.reset(r.page.run); await settle();
  r.client.points.delete(6); r.client.line = r.client.line.filter(p => p.seq !== 6);   // thinned out of the series
  r.page.seq = 7; r.client.frame();
  assert.equal(r.view().summary.seq, 4, 'never blank, never later'); assert.ok(r.calls.some(c => c[0] === 'summary' && c[1] === 6));
  await settle(); assert.equal(r.view().summary.seq, 6);
  r.page.seq = 30; r.client.frame(); r.client.frame(); await settle(); r.client.frame();
  assert.equal(r.calls.filter(c => c[0] === 'summary' && c[1] === 30).length, 0);
  r.calls.length = 0; r.client.missing.add(5); r.client.fetchSummary(5); assert.equal(r.calls.length, 0, 'a 404 is not asked again on every render');
});

test('whole samples (20,000 cells or more) apply at most one result a second while following live', async () => {
  const r = rig({ cells: 41190 }); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.client.frame(); assert.equal(r.view().summary.seq, 18);
  r.advance(300); r.client.event(ready(19)); r.page.seq = 21; r.client.frame();
  assert.equal(r.view().summary.seq, 18, 'held: the previous one was applied 0.3 s ago'); assert.equal(r.timers.size >= 1, true);
  r.advance(COALESCE_MS - 300); r.flush(); assert.equal(r.view().summary.seq, 19);
  const small = rig({ cells: 3000 }); small.page.seq = 20; small.client.reset(small.page.run); await settle();
  small.client.event(ready(18)); small.client.frame(); small.advance(100); small.client.event(ready(19)); small.client.frame();
  assert.equal(small.view().summary.seq, 19, 'a crop applies every result');
});

test('with Mixing colours a live result waits for its values, so the number and the colours change together', async () => {
  const r = rig({ wantCells: true }); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.client.frame();
  assert.equal(r.view().summary.seq, 18, 'the first result shows at once'); assert.equal(r.view().cells, null);
  await settle(); r.client.frame(); assert.deepEqual([...r.view().cells], [80, 255, 20]);
  r.client.event(ready(19)); r.client.frame(); assert.equal(r.view().summary.seq, 18, 'its values are on their way');
  await settle(); r.client.frame(); assert.equal(r.view().summary.seq, 19); assert.ok(r.view().cells);
  assert.equal(r.calls.filter(c => c[0] === 'cells' && c[1] === 19).length, 1);
  assert.deepEqual(r.calls.find(c => c[0] === 'cells'), ['cells', 18, 'm1', 'averaged'], 'the URL names the model and the count: cached as immutable');
});

test('a cell\'s value and its start value load on demand for the tooltip and the card', async () => {
  const r = rig({ live: false }); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18));
  assert.equal(r.client.cellValue(1), null, 'asked for'); await settle();
  assert.equal(r.client.cellValue(1), 80); assert.equal(r.client.cellValue(2), 255); assert.equal(r.client.cellValue(9), null);
  assert.equal(r.client.startValue(1), null); await settle(); assert.equal(r.client.startValue(1), 3); assert.equal(r.view().startCells, true);
  r.page.seq = 0; r.client.frame();
  assert.equal(r.view().atStart, true); assert.equal(r.view().summary, null); assert.deepEqual([...r.view().cells], [3, 255, 20], 'frame 0 shows the start line');
});

test('learning: the sample\'s events carry progress and reasons; ready asks for the run again', async () => {
  const r = rig({ describe: { state: 'waiting', reason: 'queue', sample_state: 'waiting', model: null, start: null } });
  r.page.seq = 3; r.client.reset(r.page.run); await settle();
  assert.equal(r.view().status, 'waiting'); assert.equal(r.view().reason, 'queue');
  r.client.event({ ev: 'qc', sample: 'other', state: 'learning' }); assert.equal(r.view().status, 'waiting', 'another sample');
  r.client.event({ ev: 'qc', sample: 's1', state: 'waiting', reason: 'lock' }); assert.equal(r.view().reason, 'lock');
  r.client.event({ ev: 'qc', sample: 's1', state: 'learning', progress: { step: 12, of: 50 }, eta_s: 80, shared: true, relearn: true });
  assert.deepEqual([r.view().status, r.view().progress.step, r.view().eta_s, r.view().shared, r.view().relearn], ['learning', 12, 80, true, true]);
  r.page.live = false; r.page.seq = 2; r.client.frame(); r.advance(REQUEST_MS);
  assert.deepEqual(r.calls.filter(c => c[0] === 'post'), [['post', { seq: 2 }]], 'asked while learning: the engine checks it first once ready');
  r.client.frame(); assert.equal(r.view().requested, true, 'and the panel says so');
  Object.assign(r.answer, { state: 'starting', sample_state: 'ready', model: { id: 'm1' } });
  r.client.event({ ev: 'qc', sample: 's1', state: 'ready' }); await settle();
  assert.equal(r.calls.filter(c => c === 'describe').length, 2); assert.equal(r.view().status, 'waiting-start');
  r.client.event({ ev: 'qc', run: 'r1', state: 'start', summary: { per100: 9.5 } }); assert.equal(r.view().status, 'scrubbing');
  r.client.event({ ev: 'qc', sample: 's1', state: 'failed', reason: 'too_few_cells' }); assert.deepEqual([r.view().status, r.view().reason], ['failed', 'too_few_cells']);
  await r.client.retry(); assert.ok(r.calls.includes('learn')); assert.equal(r.view().status, 'waiting');
});

test('the switch shows at once and is put back with a notice if the engine refuses; off forgets what was shown', async () => {
  const r = rig(); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.client.frame(); assert.ok(r.view().summary);
  r.rigState.refuse = true;
  const pending = r.client.setEnabled(false);
  assert.equal(r.view().status, 'off'); assert.equal(r.view().summary, null);
  await pending; assert.equal(r.view().enabled, true);
  assert.equal(r.notices.at(-1), "Couldn't switch the mixing check for this run. Try again.", "in plain words: never the engine's own message");
  r.rigState.refuse = false; await r.client.setEnabled(false); await settle();
  assert.equal(r.view().status, 'off'); assert.deepEqual(r.view().areas, []);
  await r.client.setEnabled(true); await settle(); assert.equal(r.view().enabled, true);
  assert.deepEqual(r.calls.filter(c => c[0] === 'post').map(c => c[1]), [{ enabled: false }, { enabled: false }, { enabled: true }]);
});

test('a relearned model\'s results never join the old line; an error says the worker restarts until the next result', async () => {
  const r = rig({ describe: { seqs: [3], latest_seq: 3 } }); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  assert.equal(r.client.line.length, 1);
  r.client.event({ ev: 'qc', run: 'r1', state: 'error', seq: 4, error: 'worker exited' }); r.client.frame();
  assert.equal(r.view().restarting, true);
  r.client.event({ ...ready(5), summary: summary(5, { model: 'm2' }) }); r.client.frame();
  assert.equal(r.client.model, 'm2'); assert.deepEqual(r.client.line.map(p => p.seq), [5]); assert.equal(r.view().restarting, false);
  for (let s = 6; s < 6 + KEEP_SUMMARIES + 10; s++) r.client.addSummary(summary(s, { model: 'm2' }));
  assert.equal(r.client.summaries.size, KEEP_SUMMARIES);
});

test('motion is allowed only for a change at least MOTION_GAP_MS after the previous one', async () => {
  const r = rig({ live: true }); r.page.seq = 30; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.client.frame(); assert.equal(r.view().motion, false, 'the first result lands');
  r.advance(100); r.client.event(ready(19)); r.client.frame(); assert.equal(r.view().motion, false, 'a crop\'s ten a second');
  r.advance(MOTION_GAP_MS); r.client.event(ready(20)); r.client.frame(); assert.equal(r.view().motion, true, 'a whole sample\'s one a second');
  r.client.frame(); assert.equal(r.view().motion, false, 'a render of the same result does not move again');
});

test('paused, a result that arrives for the frame on screen lands without motion; a scrub moves once', async () => {
  const r = rig({ live: false }); r.page.seq = 30; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.advance(MOTION_GAP_MS);
  r.client.event(ready(20));
  assert.equal(r.view().summary.seq, 20); assert.equal(r.view().motion, false, 'a late result while paused: still');
  assert.equal(r.view().still, true);
  r.advance(MOTION_GAP_MS); r.page.seq = 19; r.client.frame();
  assert.equal(r.view().summary.seq, 18); assert.equal(r.view().motion, true, 'the transition a scrub causes');
  assert.equal(r.view().still, false);
  r.advance(MOTION_GAP_MS); r.client.event(ready(19));
  assert.equal(r.view().summary.seq, 19); assert.equal(r.view().motion, true, 'its own result, asked for: part of the scrub');
  r.advance(MOVE_WINDOW_MS); r.client.event(ready(19, { per100: 12 }));
  assert.equal(r.view().motion, false, 'scored again later: still'); assert.equal(r.view().still, true);
});

const startArea = { L: 'A', tag: 'already', per100: 18.3, start_per100: 18.3, judged: 491, outline: [[0, 0], [100, 0], [100, 100]] };

test("the start line's own areas are kept apart: an answer without them never drops them; frame 0 and the frames before cells form hold them", async () => {
  const r = rig({ live: false }); r.page.seq = 0; r.page.run.state = 'paused'; r.page.run.ack.state = 'paused';
  // The WebSocket's start event (with its areas) comes first, the run's answer (numbers only) after it.
  r.client.reset(r.page.run);
  r.client.event({ ev: 'qc', run: 'r1', state: 'start', summary: { per100: 11.8, judged: 3200, mixed: 379, median_tx: 386, areas: [startArea], more_spots: 1 } });
  await settle();
  let v = r.view();
  assert.equal(v.atStart, true); assert.equal(v.areaView, 'start');
  assert.deepEqual(v.areas.map(a => a.L), ['A'], 'still there after GET /runs/{id}/qc');
  assert.equal(v.start.per100, 10, 'the numbers follow the newest answer'); assert.equal(v.start.areas, undefined, 'the areas live apart');
  assert.ok(!r.calls.includes('start'), 'nothing to ask for: the event brought them');
  // A frame while cells still form (the engine's area_view "start"): the start line's areas; from burn-in + 20
  // iterations the frame's own (here none).
  r.page.seq = 3; r.client.event(ready(3, { phase: 'burnin', areas: [], area_view: 'start' }));
  v = r.view(); assert.equal(v.summary.seq, 3); assert.equal(v.areaView, 'start'); assert.deepEqual(v.areas.map(a => a.L), ['A']);
  r.page.seq = 12; r.client.event(ready(12, { areas: [], area_view: 'smoothed' }));
  v = r.view(); assert.deepEqual(v.areas, []); assert.equal(v.areaView, 'smoothed');
  r.page.seq = 13; r.client.event(ready(13, { area_view: 'single' }));
  assert.deepEqual(r.view().areas.map(a => a.L), ['A']); assert.equal(r.view().areaView, 'single');
});

test("without an event the start line's areas are asked for once per model, when the view is to hold them", async () => {
  const r = rig({ live: false }); r.page.seq = 0; r.client.reset(r.page.run); await settle();
  assert.equal(r.calls.filter(c => c === 'start').length, 1, 'frame 0 holds them: asked for');
  assert.deepEqual(r.view().areas, [], 'a 404 (none yet): none');
  r.client.frame(); r.client.frame(); await settle(); r.client.frame();
  assert.ok(r.calls.filter(c => c === 'start').length <= 2, 'a failure is asked again at most once more, not on every render');
  r.rigState.startAnswer = { per100: 10, judged: 300, model: 'm1', areas: [startArea], more_spots: 0 };
  r.client.startAsked = null; r.client.frame(); await settle();
  assert.deepEqual(r.view().areas.map(a => a.L), ['A']);
  const asked = r.calls.filter(c => c === 'start').length;
  r.client.frame(); r.page.seq = 2; r.client.event(ready(2, { area_view: 'start', areas: [] })); await settle();
  assert.equal(r.calls.filter(c => c === 'start').length, asked, 'once known, never asked again');
  assert.deepEqual(r.view().areas.map(a => a.L), ['A']);
  // A relearned model: its start line (and areas) are new.
  r.client.event({ ...ready(5), summary: summary(5, { model: 'm2' }) }); await settle();
  assert.equal(r.client.startAreas, null);
  const nuclei = rig({ describe: { start: null, start_source: 'nuclei' }, live: false }); nuclei.page.seq = 0;
  nuclei.client.reset(nuclei.page.run); await settle();
  assert.ok(!nuclei.calls.includes('start'), 'a run that started from nuclei only has no start line');
  assert.deepEqual(nuclei.view().areas, []); assert.equal(nuclei.view().start, null);
});

test("since no area is shown the view keeps nothing that only an area list read: the start line's and the results' areas alone", async () => {
  // What an area list once needed (the spots left out of it, how many it listed, from when the run's own areas are
  // shown), should an engine still send it: none of it reaches the view, the panel or the page.
  const gone = ['moreSpots', 'listed', 'areasKnown', 'areasFrom', 'startAreas'];
  const r = rig({ describe: { area_from_iter: 120 }, live: false }); r.page.seq = 0; r.page.run.state = 'paused'; r.page.run.ack.state = 'paused';
  r.client.reset(r.page.run);
  r.client.event({ ev: 'qc', run: 'r1', state: 'start', summary: { per100: 11.8, judged: 3200, areas: [startArea], more_spots: 3, listed: 3 } });
  await settle();
  let v = r.view();
  assert.deepEqual(v.areas.map(a => a.L), ['A']); assert.deepEqual(gone.filter(key => key in v), []);
  assert.deepEqual(r.client.startAreas, [startArea], 'the start line\'s areas, nothing beside them');
  assert.equal('more_spots' in v.start || 'listed' in v.start, false, 'nor in the start line\'s numbers');
  r.page.seq = 3; r.client.event(ready(3, { phase: 'burnin', areas: [], area_view: 'start', more_spots: 2, listed: 3 }));
  v = r.view(); assert.deepEqual(v.areas.map(a => a.L), ['A']); assert.deepEqual(gone.filter(key => key in v), []);
  r.page.seq = 12; r.client.event(ready(12, { area_view: 'smoothed', more_spots: 2, listed: 3 }));
  v = r.view(); assert.deepEqual(v.areas.map(a => a.L), ['A']); assert.deepEqual(gone.filter(key => key in v), []);
});

test("frame 0 (nuclei only) never shows its own result or joins the line: the start line stands for it", async () => {
  const r = rig({ describe: { seqs: [0, 3], latest_seq: 3 }, live: false }); r.page.seq = 10;
  r.client.reset(r.page.run); await settle();
  assert.deepEqual(r.client.seqs, [3]); assert.deepEqual(r.view().line.map(p => p.seq), [3], 'the series\' frame 0 is left out');
  r.client.event(ready(0, { per100: 6.6, shown: 7 }));
  assert.deepEqual(r.view().line.map(p => p.seq), [3]); assert.equal(r.client.summaries.has(0), false);
  r.page.seq = 1; r.client.frame();
  assert.equal(r.view().summary.seq, 3, 'frame 1, never checked: the oldest after it (never the nuclei-only result)');
  assert.equal(r.view().snapped, 30);
  r.page.seq = 0; r.client.frame(); assert.equal(r.view().atStart, true); assert.equal(r.view().summary, null); assert.equal(r.view().snapped, null);
  r.page.live = true; r.page.seq = 1; r.client.frame(); assert.equal(r.view().summary, null, 'following live: a later result waits for its snapshot');
});

test("a run with no start line: from nuclei only, or from cells with their nuclei (start_kind \"paired\"); its start never shows", async () => {
  // Proseg's count at iteration 0 holds the nuclei alone in either case: the engine gives no start line, and an answer
  // that still carried one would never be shown, nor its per-cell values asked for.
  for (const describe of [{ start_kind: 'paired', start_source: 'xenium' }, { start_kind: 'nuclei', start_source: 'nuclei' }]) {
    const r = rig({ describe: { seqs: [3], latest_seq: 3, ...describe }, live: false }); r.page.seq = 0;
    r.rigState.startAnswer = { per100: 10, judged: 300, model: 'm1', areas: [startArea], more_spots: 0 };   // were it asked
    r.client.reset(r.page.run); await settle();
    assert.equal(r.view().startKind, describe.start_kind); assert.equal(r.view().start, null); assert.equal(r.view().atStart, true);
    assert.equal(r.client.startValue(1), null); assert.ok(!r.calls.some(c => c[0] === 'cells' && c[1] === 'start'), 'never its values');
    // Nor its areas: not at frame 0, nor while cells form (the engine's area_view "start"), where no area is held.
    assert.ok(!r.calls.includes('start'), 'GET /qc/start never asked'); assert.deepEqual(r.view().areas, []);
    r.page.seq = 3; r.client.event(ready(3, { phase: 'burnin', area_view: 'start' })); await settle();
    assert.equal(r.view().areaView, 'start'); assert.deepEqual(r.view().areas, []); assert.ok(!r.calls.includes('start'));
    r.page.seq = 5; r.client.frame(); assert.equal(r.view().summary.seq, 3); assert.equal(r.view().start, null);
    // The page's log names no change since a start the run does not have, whatever the result's change block says.
    r.client.summaries.set(3, { ...r.client.summaries.get(3), change: { raw: null, matched: null, threshold: 1, model_threshold: 0.6, meaningful: false,
      direction: null, enough: false } });
    r.client.frame();
    assert.match(r.client.logText(), /no change since the start: this kind of start has no start line/);
    assert.doesNotMatch(r.client.logText(), /change since the start: raw/);
  }
  // From cells alone (start_kind "cells", or an engine that names none): its start line, as before.
  for (const describe of [{ start_kind: 'cells' }, {}]) {
    const r = rig({ describe: { seqs: [3], latest_seq: 3, ...describe } }); r.page.seq = 5; r.client.reset(r.page.run); await settle();
    assert.equal(r.view().start.per100, 10); assert.equal(r.view().startKind, describe.start_kind ?? null);
  }
});

test('a result known only by its point keeps the earlier result shown until its areas arrive; a point has its frame\'s phase', async () => {
  const r = rig({ describe: { seqs: [2, 4, 6], latest_seq: 6 }, live: false }); r.page.seq = 2;
  r.client.reset(r.page.run); await settle(); await settle();
  assert.equal(r.view().summary.seq, 2); assert.ok(r.client.summaries.has(2), 'its summary was fetched');
  // Scrubbing forward: frame 4 is known by its point only; its summary is on its way.
  let release; const gate = new Promise(resolve => { release = resolve; });
  const fetch = r.client.api.qcSummary;
  r.client.api.qcSummary = async (id, seq) => { if (seq === 4) await gate; return fetch(id, seq); };
  r.page.seq = 4; r.client.frame();
  assert.equal(r.view().summary.seq, 2, 'number and areas stay those of frame 2 for now (one change, not two)');
  release(); await settle();
  assert.equal(r.view().summary.seq, 4); assert.deepEqual(r.view().areas.map(a => a.L), ['A']);
  // Back to an earlier frame known only by its point: never the later one; its point alone until its summary.
  r.client.summaries.delete(2); r.client.api.qcSummary = async () => new Promise(() => {});
  r.page.seq = 3; r.client.frame();
  assert.equal(r.view().summary.seq, 2); assert.equal(r.view().summary.areas, undefined);
  assert.equal(r.view().summary.phase, 'burnin', 'a point has no phase of its own: its frame\'s');
});

test("a run that started from nuclei only never waits for a start line", () => {
  const base = { describe: { state: 'ready', sample_state: 'ready' }, sample: null, enabled: true, results: false, start: null,
    run: { state: 'paused', ack: { state: 'paused' } }, seq: 0, newest: 0, live: false };
  assert.equal(panelStatus(base), 'paused', 'ready without a start line: none is coming');
  assert.equal(panelStatus({ ...base, describe: { state: 'starting', sample_state: 'ready' } }), 'waiting-start');
});

test('per-cell values that cannot be fetched never stall the live number, never repeat on every render, and keep the earlier colours', async () => {
  const r = rig({ wantCells: true }); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.client.frame(); await settle(); r.client.frame();
  assert.deepEqual([...r.view().cells], [80, 255, 20]);
  const fetch = r.client.api.qcCells;
  r.client.api.qcCells = async (id, seq, model) => { r.calls.push(['cells', seq, model]); if (seq === 19) throw Object.assign(new Error('gone'), { status: 404 }); return fetch(id, seq, model); };
  r.client.event(ready(19)); r.client.frame(); await settle(); r.client.frame();
  assert.equal(r.view().summary.seq, 19, 'the number moves on without its values');
  assert.deepEqual([...r.view().cells], [80, 255, 20], "the cells keep frame 18's colours (an earlier result, never a later one)");
  for (let i = 0; i < 5; i++) r.client.frame();
  await settle();
  assert.equal(r.calls.filter(c => c[0] === 'cells' && c[1] === 19).length, 1, 'asked once, not on every render');
  r.client.refresh(); await settle(); r.client.frame(); await settle();
  assert.equal(r.calls.filter(c => c[0] === 'cells' && c[1] === 19).length, 2, 'a reconnect asks again');
});


/** A rig whose per-cell values arrive only when the test releases them: values(seq) resolves frame seq's (of every
 *  count asked for), values(seq, count) those of one count, values(seq, count, error) fails them. Each count has its own
 *  bytes: the first is seq on the averaged count, 100 + seq on the count proseg saves ("sampled"). */
function slowCells(r) {
  const gates = new Map();
  r.client.api.qcCells = (id, seq, model, signal, count) => {
    r.calls.push(['cells', seq, model]);
    return new Promise((resolve, reject) => gates.set(`${seq}:${count}`, error => error ? reject(error)
      : resolve(Uint8Array.of(seq === 'start' ? 3 : count === 'sampled' ? (100 + seq) % 250 : seq % 250, 255, 20))));
  };
  return async (seq, count, error) => {
    for (const [key, release] of [...gates]) if (count === undefined ? key.startsWith(`${seq}:`) : key === `${seq}:${count}`) { gates.delete(key); release(error); }
    await settle();
  };
}

test('paused, stepped or scrubbed with Mixing colours: the cells never go grey between two results, and a step moves', async () => {
  const r = rig({ live: false, wantCells: true }); r.page.run.state = 'paused'; r.page.run.ack.state = 'paused';
  const values = slowCells(r);
  r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); await values(18); r.client.frame();
  assert.deepEqual([...r.view().cells], [18, 255, 20]);
  const seen = [];
  const render = r.client.renderView; r.client.renderView = v => { seen.push(v.cells ? v.cells[0] : null); render(v); };
  // The paused frame's own result lands: until its values arrive, frame 18's result stays (number and colours).
  r.advance(MOTION_GAP_MS); r.client.event(ready(20));
  assert.equal(r.view().summary.seq, 18, 'held: the number changes with its colours'); assert.equal(r.view().cells[0], 18);
  await values(20);
  assert.equal(r.view().summary.seq, 20); assert.equal(r.view().cells[0], 20);
  assert.equal(r.view().still, true, 'a late result for the frame on screen lands without motion');
  // A step (the frame on screen moves) whose result and values come a moment later: still no grey, and it may move.
  r.advance(MOTION_GAP_MS); r.page.seq = 21; r.client.frame();
  assert.equal(r.view().summary.seq, 20);
  r.client.event(ready(21)); await values(21);
  assert.equal(r.view().summary.seq, 21); assert.equal(r.view().still, false, 'the transition the step causes');
  assert.equal(r.view().motion, true);
  // Back to an earlier frame whose values are not loaded: the newest values at or before it (here the start line's,
  // never a later frame's) until its own arrive.
  r.client.cells.set('start', Uint8Array.of(3, 255, 20));
  r.page.seq = 19; r.client.frame();
  assert.equal(r.view().summary.seq, 18); assert.notEqual(r.view().cells, null);
  assert.ok([18, 3].includes(r.view().cells[0]), 'an earlier frame\'s values, never 20\'s or 21\'s');
  assert.ok(!seen.includes(null), `no render without values: ${seen}`);
});

test('with Mixing colours a fast run applies its results as they come: values are asked for when each result arrives', async () => {
  const r = rig({ live: true, wantCells: true }); r.page.seq = 1; r.client.reset(r.page.run); await settle();
  for (let k = 1; k <= 30; k++) {
    r.page.seq = k; r.client.frame(); r.flush();              // snapshot k (its paint batch)
    r.client.event(ready(k)); await settle();                 // its result arrives (its values with it), before k + 1
  }
  assert.ok(r.counts.qcApplied >= 28, `applied ${r.counts.qcApplied} of 30`);
  assert.equal(r.view().summary.seq, 29, 'the newest at or before the frame on screen (30\'s comes with 31)');
  assert.ok(r.view().cells, 'with its values');
  assert.equal(r.calls.filter(c => c[0] === 'cells' && c[1] === 12).length, 1, 'each asked for once');
  // Paused, only the result of the frame on screen is asked for ahead (not a backfilled one).
  const p = rig({ live: false, wantCells: true }); p.page.seq = 20; p.client.reset(p.page.run); await settle();
  p.client.event(ready(20)); p.client.event(ready(7)); await settle();
  assert.deepEqual(p.calls.filter(c => c[0] === 'cells').map(c => c[1]), [20]);
});

test("an old error replayed by the engine never leaves 'the worker restarts' up; a fresh one does until the next result", async () => {
  const r = rig({ describe: { seqs: [30, 31, 32, 33], latest_seq: 33 }, live: false });
  r.page.run.state = 'done'; r.page.seq = 33; r.client.reset(r.page.run); await settle();
  for (const s of [30, 31, 32, 33]) r.client.event(ready(s));
  r.client.event({ ev: 'qc', run: 'r1', state: 'error', seq: 12, error: 'worker exited' });   // answered long ago
  assert.equal(r.view().restarting, false);
  r.client.event({ ev: 'qc', run: 'r1', state: 'error', seq: null, error: 'start line failed' });   // the start line exists
  assert.equal(r.view().restarting, false);
  r.client.event({ ev: 'qc', run: 'r1', state: 'error', seq: 34, error: 'worker exited' });
  assert.equal(r.view().restarting, true, 'newer than every result');
  Object.assign(r.answer, { scoring: { state: 'error', seq: 20 } });
  r.client.refresh(); await settle();
  assert.equal(r.view().restarting, false, 'a reconnect asks the scheduler: that failure was answered');
  Object.assign(r.answer, { scoring: { state: 'error', seq: 40 } });
  r.client.refresh(); await settle(); assert.equal(r.view().restarting, true);
  r.client.event({ ev: 'qc', run: 'r1', state: 'start', summary: { per100: 10 } }); assert.equal(r.view().restarting, false);
});

test('a summary that failed for another reason than "not checked yet" is asked again; its areas come with it', async () => {
  const r = rig({ describe: { seqs: [5, 6], latest_seq: 6 }, live: false });
  r.page.run.state = 'done'; r.page.seq = 6;
  const fetch = r.client.api.qcSummary; let down = true;
  r.client.api.qcSummary = async (id, seq) => {
    if (down && seq === 6) { r.calls.push(['summary', seq]); throw new TypeError('Failed to fetch'); }
    return fetch(id, seq);                                  // (records the call itself)
  };
  r.client.reset(r.page.run); await settle(); r.client.frame(); await settle();
  assert.equal(r.view().summary.seq, 6); assert.deepEqual(r.view().areas, [], 'its point only: no areas yet');
  r.client.frame(); await settle();
  assert.equal(r.calls.filter(c => c[0] === 'summary' && c[1] === 6).length, 1, 'not again at once');
  down = false; r.advance(RETRY_MS[0]); r.client.frame(); await settle();
  assert.equal(r.calls.filter(c => c[0] === 'summary' && c[1] === 6).length, 2, 'asked again after a short wait');
  assert.deepEqual(r.view().areas.map(a => a.L), ['A']);
  r.client.missing.add(5); r.client.refresh(); await settle();
  assert.equal(r.client.missing.has(5), false, 'a reconnect forgets what was not checked yet');
});

test("results the engine only named (a busy tick) and the frame asked for's answer are known; a run's answer is asked again after a failure", async () => {
  const r = rig({ describe: { seqs: [3], latest_seq: 3 }, live: false }); r.page.seq = 9; r.client.reset(r.page.run); await settle();
  r.client.event({ ev: 'qc', run: 'r1', state: 'seqs', seqs: [4, 5, 0] });
  assert.deepEqual(r.client.seqs, [3, 4, 5]);
  r.client.api.setRunQc = async (id, body) => { r.calls.push(['post', body]); return { ...r.answer, seqs: [3, 4, 5, 7, 9] }; };
  r.client.frame(); r.advance(REQUEST_MS); await settle();
  assert.deepEqual(r.calls.filter(c => c[0] === 'post').at(-1), ['post', { seq: 9 }]);
  assert.deepEqual(r.client.seqs, [3, 4, 5, 7, 9], 'the answer names frame 9: its result is fetched');
  const down = rig({ describe: { seqs: [3] } });
  let fail = true; const answer = down.client.api.runQc;
  down.client.api.runQc = async (id, signal) => { down.calls.push('describe'); if (fail) throw new TypeError('Failed to fetch'); return answer(id, signal); };
  down.page.seq = 5; down.client.reset(down.page.run); await settle();
  assert.equal(down.view().status, 'loading'); assert.equal(down.view().failed, true);
  fail = false; down.advance(RETRY_MS[0]); await settle();
  assert.equal(down.view().failed, false); assert.equal(down.view().status, 'live');
});

test("a long run's points stay bounded; the forming band ends at the newest point, not at every snapshot", async () => {
  const r = rig({ live: true }); r.page.seq = 0; r.client.reset(r.page.run); await settle();
  for (let s = 1; s <= TRIM_POINTS + 50; s++) r.client.addPoint({ seq: s, iter: s, per100: 10, shown: 10, judged: 300 });
  assert.ok(r.client.points.size <= KEEP_POINTS + r.client.line.length + KEEP_SUMMARIES, `${r.client.points.size} points`);
  assert.equal(r.client.seqs.length, TRIM_POINTS + 50, 'every result is still known');
  const f = rig({ live: true }); f.page.seq = 3; f.client.reset(f.page.run); await settle();
  f.client.event(ready(2, { phase: 'burnin' })); f.client.frame();
  const until = f.view().formingUntil;
  f.page.seq = 4; f.client.frame();
  assert.equal(f.view().formingUntil, until, 'a new snapshot while cells form does not move it');
  assert.equal(f.view().runState, 'running');
});

// ---- the card, the trial and the previous run ---------------------------------------------------------------------------
const card = (extra = {}) => ({ id: 'R4:cell-compactness:0.02', rule: 'R4', areas: ['A'], flag: 'cell-compactness', from: 0.04, to: 0.02, confidence: 'rough',
  timing: 'later', actions: ['new_run', 'try', 'why'], main: 'new_run', ...extra });
const live = async (r, seq, extra = {}) => { r.answer.seqs.push(seq); r.client.event(ready(seq, extra)); r.page.seq = seq; r.flush(); await settle(); return r.view(); };

test("the result shown carries the engine's card and its more ideas; a card answered here never comes back for this run", async () => {
  const r = rig({ describe: { seqs: [] } }); r.client.reset(r.page.run); await settle();
  let view = await live(r, 10, { suggestion: card(), more_ideas: [card({ id: 'R5', rule: 'R5', flag: 'prior-weight', from: 1, to: 2 })] });
  assert.equal(view.suggestion.rule, 'R4'); assert.deepEqual(view.moreIdeas.map(s => s.rule), ['R5']);
  view = await live(r, 11, { suggestion: null, more_ideas: [] });
  assert.equal(view.suggestion, null, 'the engine keeps its card steady: the page shows what it says');
  const undo = { id: 'U:cell-compactness:0.04', rule: 'U', areas: [], flag: 'cell-compactness', from: 0.02, to: 0.04 };
  view = await live(r, 12, { suggestion: undo });
  assert.equal(view.suggestion.rule, 'U');
  await r.client.answerUndoCard(undo, 'kept');
  assert.deepEqual(r.calls.at(-1), ['end', 'kept', 'U'], 'the Undo card is answered with rule U');
  view = await live(r, 13, { suggestion: undo }); assert.equal(view.suggestion, null, 'answered here: not shown again');
  r.rigState.endStatus = 404; await r.client.answerUndoCard({ ...undo, to: 0.05 }, 'undone');   // the engine had dropped it already: no error
  r.rigState.endStatus = 503; await assert.rejects(r.client.answerUndoCard({ ...undo, to: 0.06 }, 'kept'));
  assert.equal(cardKey(undo), cardKey({ ...undo, from: 1 }), 'a card is the engine\'s id: its rule and change');
  // A refusal leaves the card offered (its answer can be given again); it goes once the engine took the answer. An answer
  // for a run no longer open is never sent.
  const later = { ...undo, id: 'U:cell-compactness:0.04:2' };
  view = await live(r, 14, { suggestion: later });
  await assert.rejects(r.client.answerUndoCard(later, 'kept'));
  view = await live(r, 15, { suggestion: later }); assert.equal(view.suggestion?.id, later.id, 'still offered after a refusal');
  r.rigState.endStatus = null;
  const sent = r.calls.length; await r.client.answerUndoCard(later, 'kept', 'another-run');
  assert.equal(r.calls.length, sent, 'not for another run'); assert.equal(await r.client.endTrial('kept', 'another-run'), null);
  await r.client.answerUndoCard(later, 'kept');
  view = await live(r, 16, { suggestion: later }); assert.equal(view.suggestion, null, 'answered: gone');
  // An earlier frame shows that frame's card; frame 0 and a switched-off check none.
  r.page.live = false; r.page.seq = 10; r.client.frame(); assert.equal(r.view().suggestion.rule, 'R4');
  r.page.seq = 0; r.client.frame(); assert.equal(r.view().suggestion, null);
});

test('Try it registers the trial; its rows come with the results; Keep ends it and its last words stay a few iterations', async () => {
  const r = rig({ describe: { seqs: [] } }); r.client.reset(r.page.run); await settle();
  await live(r, 20, { suggestion: card() });
  const trial = await r.client.startTrial({ rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02 });
  assert.deepEqual(r.calls.find(c => c[0] === 'trial'), ['trial', { rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02 }]);
  assert.equal(trial.id, 1); assert.equal(r.view().trial.id, 1, 'shown at once, before any result'); assert.equal(r.counts.qcTrials, 1);
  // Asked for at iteration 200 while the result shown is still frame 20's (iteration 200 or before): shown at once all the
  // same, so the card goes with the click (a second click would only be refused).
  r.client.trial.asked_iter = 250; r.client.frame();
  assert.equal(r.view().trial.id, 1, 'the newest frame shows the page\'s own trial'); assert.equal(r.view().suggestion.rule, 'R4', 'the panel hides the card');
  // The results bring the trial as of each result: its start, rows and verdict (counted as the verdict changes). A
  // summary's copy is compact (no asked_iter, no started): what the page knew of it stays.
  const rows = [{ kind: 'whole', expected: 11.2, now: 10.6 }];
  const { asked_iter: _a, started: _s, ...compact } = trial;
  let view = await live(r, 21, { trial: { ...compact, start_seq: 21, start_iter: 210, rows, verdict: null } });
  assert.equal(view.trial.start_iter, 210); assert.deepEqual(view.trial.rows, rows); assert.equal(view.trial.asked_iter, 250, 'kept from the page\'s own copy');
  view = await live(r, 22, { trial: { ...trial, start_seq: 21, start_iter: 210, rows, verdict: 'less' } });
  assert.equal(view.trial.verdict, 'less'); assert.equal(r.counts.qcVerdicts, 1);
  r.client.event(ready(19, { trial: { ...trial, verdict: null } })); r.flush(); await settle();   // a late, older result
  assert.equal(r.client.trial.verdict, 'less', 'an older result never undoes a newer answer');
  await r.client.endTrial('kept');
  assert.deepEqual(r.calls.at(-1), ['end', 'kept', null]);
  view = r.view(); assert.equal(view.trial, null); assert.equal(view.trialEnded.outcome, 'kept');
  // A result sent before the end still carries it: it never comes back; the ended note goes after ENDED_NOTE_ITERS.
  view = await live(r, 23, { trial: { ...trial, verdict: 'less' } }); assert.equal(view.trial, null); assert.equal(view.trialEnded.outcome, 'kept');
  r.client.event({ ev: 'qc', run: 'r1', state: 'trial', trial: { ...trial, verdict: 'less' }, n: 4 }); assert.equal(r.client.trial, null, 'nor a push sent before it');
  const newest = r.page.frames.at(-1).iter; r.client.trialEnded.endedIter = newest - ENDED_NOTE_ITERS; r.client.frame();
  assert.equal(r.view().trialEnded, null, 'gone after a few iterations');
});

test('a trial started, kept or undone elsewhere follows the engine; a refused one throws; an earlier frame shows its own copy', async () => {
  const r = rig({ describe: { seqs: [] } }); r.client.reset(r.page.run); await settle();
  await live(r, 20, {});
  const other = { id: 7, rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02, asked_iter: 195, start_iter: 200, rows: [], verdict: null };
  r.client.event({ ev: 'qc', run: 'r1', state: 'trial', trial: other, n: 1 });
  assert.equal(r.view().trial.id, 7, 'another window tried it');
  r.client.event({ ev: 'qc', run: 'r1', state: 'trial', trial: null, n: 2 });
  assert.equal(r.view().trial, null, 'and kept or undid it');
  r.rigState.trialRefused = true;
  await assert.rejects(r.client.startTrial({ rule: 'R4', area: 'A', flag: 'cell-compactness', from: 0.04, to: 0.02 }), /already running/);
  assert.equal(r.client.trial, null);
  // The run's answer is the trial now (a reconnect): it starts or ends one the page knew nothing of.
  r.answer.trial = other; await r.client.refresh(); assert.equal(r.client.trial.id, 7);
  r.answer.trial = null; await r.client.refresh(); assert.equal(r.client.trial, null);
  // Scrubbing: an earlier result shows its own copy (read only), never one from before the trial was asked for.
  r.answer.trial = other; await r.client.refresh();
  await live(r, 30, { trial: { ...other, verdict: 'more' } });
  r.page.live = false; r.page.seq = 25; r.answer.seqs.push(25); r.client.event(ready(25, { trial: { ...other, verdict: null, rows: [{ kind: 'whole', expected: 9, now: 9.1 }] } }));
  r.client.frame(); assert.equal(r.view().trial.verdict, null, 'that frame\'s own copy'); assert.equal(r.view().trial.rows[0].now, 9.1);
  r.page.seq = 19; r.client.frame(); assert.equal(r.view().trial, null, 'a frame from before it was asked for (iteration 190 < 195)');
  // A 404 on Keep: it had ended already.
  r.page.live = true; r.page.seq = 30; r.rigState.endStatus = 404; await r.client.endTrial('kept');
  assert.equal(r.client.trial, null); assert.equal(r.view().trialEnded.outcome, 'kept');
  r.rigState.endStatus = 503; r.answer.trial = { ...other, id: 8 }; await r.client.refresh();
  await assert.rejects(r.client.endTrial('undone')); assert.equal(r.client.trial.id, 8, 'a failure keeps it running');
});

test("a cell readout never shows an earlier result's value as the result shown's; it says which result a value is from", async () => {
  // The case: the summary of frame 10 shown while the only values the page has are frame 2's (Mixing colours on:
  // they keep colouring the cells, since frame 10's cannot come).
  const r = rig({ live: false, wantCells: true }); r.page.run.state = 'paused'; r.page.run.ack.state = 'paused';
  r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.page.seq = 2; r.client.event(ready(2)); await settle(); r.client.frame();
  assert.equal(r.view().summary.seq, 2); assert.deepEqual([r.view().cellsSeq, r.view().cellsIter, r.view().cellsCurrent], [2, 20, true]);
  assert.equal(r.client.cellValue(1), 80); assert.deepEqual(r.client.cellReading(1), { code: 80, seq: 2, iter: 20, current: true });
  const fetch = r.client.api.qcCells;
  r.client.api.qcCells = async (id, seq, model) => {
    if (seq !== 10) return fetch(id, seq, model);
    r.calls.push(['cells', seq, model]); throw Object.assign(new Error('gone'), { status: 404 });
  };
  r.page.seq = 10; r.client.event(ready(10)); await settle(); r.client.frame();
  assert.equal(r.view().summary.seq, 10, 'the number moves on'); assert.deepEqual([...r.view().cells], [80, 255, 20], "frame 2's colours meanwhile");
  assert.deepEqual([r.view().cellsSeq, r.view().cellsIter, r.view().cellsCurrent], [2, 20, false]);
  assert.equal(r.client.cellValue(1), null, "never frame 2's value as frame 10's (it was 200 from frame 2 before)");
  const reading = r.client.cellReading(1);
  assert.deepEqual(reading, { code: 80, seq: 2, iter: 20, current: false }, 'said with the result it is from');
  assert.match(cardLine(reading.code, null, 'xenium', { iter: reading.iter }), /^Mixing at iteration 20: the model read about 32 % /);
  assert.equal(r.client.cellReading(99), null, 'a label past the values');
  // Frame 0 shows the start line's own values: current there.
  r.client.cells.set('start', Uint8Array.of(3, 255, 20)); r.page.seq = 0; r.client.frame();
  assert.deepEqual([r.view().cellsSeq, r.view().cellsIter, r.view().cellsCurrent], ['start', null, true]); assert.equal(r.client.cellValue(1), 3);
});

test("without Mixing colours a readout's values are the result shown's own: asked for on demand, then with each new result, which waits for them", async () => {
  const r = rig({ live: true }); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.client.frame();
  assert.equal(r.view().summary.seq, 18); assert.equal(r.view().cells, null, 'no colours: no values, and never an earlier result\'s');
  assert.equal(r.calls.filter(c => c[0] === 'cells').length, 0, 'nothing asked for until a readout needs it');
  assert.equal(r.client.cellValue(1), null, 'asked for'); assert.ok(r.calls.some(c => c[0] === 'cells' && c[1] === 18));
  await settle(); r.flush(); r.client.frame();
  assert.equal(r.client.cellValue(1), 80); assert.equal(r.view().cellsCurrent, true);
  // A readout in use: the next result's values are asked for as it arrives, and it waits for them (the card never blinks).
  const values = slowCells(r);
  r.client.event(ready(19)); r.page.seq = 21; r.client.frame();
  assert.ok(r.calls.some(c => c[0] === 'cells' && c[1] === 19), 'asked for as it arrives');
  assert.equal(r.view().summary.seq, 18, 'held until its values are in'); assert.equal(r.client.cellValue(1), 80);
  await values(19); r.flush(); r.client.frame();
  assert.equal(r.view().summary.seq, 19); assert.equal(r.client.cellValue(1), 19);
  // Once no readout has asked for READOUT_MS, results neither wait for their values nor ask for them.
  r.advance(READOUT_MS); r.flush();
  r.client.event(ready(20)); r.flush(); r.page.seq = 22; r.client.frame();
  assert.equal(r.view().summary.seq, 20); assert.equal(r.calls.filter(c => c[0] === 'cells' && c[1] === 20).length, 0);
  assert.equal(r.view().cells, null);
});

test("the Log button's 'Mixing detail': coverage, retention, the raw and like-for-like change, the previous run, the model, and where the cell values are from", async () => {
  const provided = new Map(), before = globalThis.sopaCompanionLog;
  globalThis.sopaCompanionLog = { provide: (name, fn) => provided.set(name, fn) };
  try {
    const provenance = { table: 'xenium_table', other_table: 'table', priors: 'single', epochs_run: 429, epochs_notebook: 50, cells: 3455, notebook_equivalent: false,
      differences: ['more_epochs_small_table', 'single_table_priors'] };
    const r = rig({ live: false, wantCells: true });
    assert.equal(typeof provided.get('Mixing detail'), 'function', 'handed to the page log when made');
    assert.equal(provided.get('Mixing detail')(), '', 'no run open');
    r.page.seq = 20; r.client.reset(r.page.run); await settle();
    r.client.event(ready(18, { coverage: { cells: 3439, judged: 3188, empty: 0, under20: 5, under100: 246, judged_share: 0.927 },
      retention: { ratio: 1.2291, tx: 1229100, start_tx: 1000000 }, change: { raw: -0.108, matched: -0.946, threshold: 0.7, model_threshold: 0.6, meaningful: true, direction: 'fewer' },
      matched: { paired: 3170, used: 3170, per100_now: 10.4, per100_start: 11.346, change: -0.946, sd: null, seeds: 0 },
      prev_change: { matched: -0.2, threshold: 1, meaningful: false, direction: 'same' }, provenance, basis: 'proseg', count: 'averaged',
      constants: { mode: 'region', id: 'outline-2026-10', measured: false } }));
    await settle(); r.client.frame();
    const text = provided.get('Mixing detail')();
    // Which count it read and which thresholds' set; like for like is worked out once, without draws (no spread).
    for (const part of ['result of iteration 180 (smoothed)', "read on proseg's count (averaged)",
      "thresholds outline-2026-10 for region runs, not re-measured on proseg's count", 'another iteration than the one on screen (iteration 200)',
      'like for like over 3,170 cells: 10.4 now, 11.346 at the start;',
      'judged 3,188 of 3,439 cells (0 empty, 5 with 1-19 transcripts, 246 with 20-99)',
      'transcripts kept 1.229 (1,229,100 now, 1,000,000 at the start)', 'change since the start: raw -0.108, like for like -0.946 (threshold 0.7, model 0.6, meaningful, fewer)',
      'model learned from xenium_table with table, single starting estimates, 429 passes (reference 50), 3,455 cells, not the reference settings (more_epochs_small_table, single_table_priors)',
      'cell values from iteration 180']) assert.ok(text.includes(part), `${part} in ${text}`);
    assert.ok(!text.includes('seeds'), text);
    r.client.enabled = false; r.client.frame(); assert.equal(provided.get('Mixing detail')(), 'off');
  } finally { if (before === undefined) delete globalThis.sopaCompanionLog; else globalThis.sopaCompanionLog = before; }
});

// ---- the count a result reads (decision A) and the rest's own numbers (H2) ------------------------------------------
test("a result counted again for the hold, a finish or the end (another count) drops the values held of the other count and asks for its own, naming it", async () => {
  // A token counted frame 18 (its result on the averaged count); the hold counted it again: the engine scores it anew on
  // the count proseg saves there, and its per-cell values are another file (GET /qc/18/cells?model=m1&count=sampled).
  const r = rig({ wantCells: true }); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18)); r.client.frame(); await settle(); r.client.frame();
  assert.deepEqual(r.calls.filter(c => c[0] === 'cells'), [['cells', 18, 'm1', 'averaged']]);
  assert.equal(r.view().line.find(p => p.seq === 18).count, 'averaged', 'points keep their count');
  r.client.event(ready(18, { count: 'sampled', single: true, per100: 12.4, shown: 12 }));
  assert.equal(r.client.cells.has(18), false, "the other count's values went");
  r.client.frame(); await settle(); r.client.frame();
  assert.deepEqual(r.calls.filter(c => c[0] === 'cells'), [['cells', 18, 'm1', 'averaged'], ['cells', 18, 'm1', 'sampled']]);
  assert.ok(r.view().cells && r.view().cellsCurrent, 'its own values shown');
  assert.equal(r.view().summary.count, 'sampled'); assert.equal(r.view().line.find(p => p.seq === 18).count, 'sampled');
  // The same count again: its values stay (asked again as any new result is: the same URL, which the browser keeps).
  r.client.event(ready(18, { count: 'sampled', single: true, per100: 12.4, shown: 12 }));
  assert.ok(r.client.cells.has(18), 'the same count again: its values stay');
  // Values of the other count on their way when the frame is counted again are dropped when they come, whichever lands
  // first; its own are asked for at once (S-UIP-2: each count its own bytes, the averaged answer last).
  const values = slowCells(r);
  r.client.event(ready(19)); r.client.frame(); await settle();
  r.client.event(ready(19, { count: 'sampled', single: true })); r.client.frame(); await settle();
  assert.deepEqual(r.calls.filter(c => c[0] === 'cells' && c[1] === 19).length, 2, 'its own asked for meanwhile');
  await values(19, 'sampled');
  assert.equal(r.client.cells.get(19)?.[0], 119, 'its own values');
  await values(19, 'averaged'); r.client.frame();
  assert.equal(r.client.cells.get(19)?.[0], 119, "a late answer of the other count never takes their place");
  assert.equal(r.view().cells?.[0], 119, 'the colours read them'); assert.equal(r.client.countAt(19), 'sampled');
  // The averaged answer first: dropped as it comes (never shown), then its own.
  r.client.event(ready(23)); r.client.frame(); await settle();
  r.client.event(ready(23, { count: 'sampled', single: true })); r.client.frame(); await settle();
  await values(23, 'averaged');
  assert.equal(r.client.cells.has(23), false, 'the other count\'s values: dropped as they come');
  await values(23, 'sampled');
  assert.equal(r.client.cells.get(23)?.[0], 123);
  // Values of the other count that fail after the frame was counted again (a 404: the engine has moved on) mark
  // nothing missing and ask for no summary again: its own values are on their way.
  r.client.event(ready(25)); r.client.frame(); await settle();
  r.client.event(ready(25, { count: 'sampled', single: true })); r.client.frame(); await settle();
  const summaries = r.calls.filter(c => c[0] === 'summary' && c[1] === 25).length;
  await values(25, 'averaged', Object.assign(new Error("This iteration's count changed"), { status: 404 })); r.client.frame(); await settle();
  assert.equal(r.client.cellsMissing.has(25), false, 'not marked missing');
  assert.equal(r.calls.filter(c => c[0] === 'summary' && c[1] === 25).length, summaries, 'no summary asked again');
  await values(25, 'sampled');
  assert.equal(r.client.cells.get(25)?.[0], 125);
});

test('a 404 for values of a count the frame no longer has asks for its summary again, once per count, and then the values of the count it reads', async () => {
  // The page holds frame 21's result on the averaged count; the hold counts it again before the push says so: values
  // asked for on the averaged count are a 404 (the engine's "This iteration's count changed").
  const r = rig({ wantCells: false }); r.page.seq = 22; r.client.reset(r.page.run); await settle();
  r.client.event(ready(21)); r.client.frame(); await settle();
  r.answer.seqs.push(21);
  r.client.api.qcSummary = async (id, seq) => { r.calls.push(['summary', seq]); return summary(seq, { count: 'sampled', single: true }); };
  r.client.api.qcCells = async (id, seq, model, signal, count) => {
    r.calls.push(['cells', seq, model, count]);
    if (count === 'averaged') throw Object.assign(new Error("This iteration's count changed"), { status: 404 });
    return Uint8Array.of(80, 255, 20);
  };
  r.page.wantCells = true;                        // Mixing colours on: its values are asked for now
  r.client.frame(); await settle(); r.client.frame(); await settle(); r.client.frame(); await settle();
  const cells = r.calls.filter(c => c[0] === 'cells').map(c => c[3]);
  assert.deepEqual(cells, ['averaged', 'sampled'], 'the 404, then its summary again, then the values of its count');
  assert.equal(r.calls.filter(c => c[0] === 'summary' && c[1] === 21).length, 1, 'its summary asked for again once');
  assert.ok(r.view().cells && r.view().cellsCurrent);
  // A count that 404s for another reason (its file gone) is not asked about in a loop.
  r.client.api.qcCells = async (id, seq, model, signal, count) => { r.calls.push(['cells', seq, model, count]); throw Object.assign(new Error('gone'), { status: 404 }); };
  r.client.event(ready(22, { count: 'sampled', single: true })); r.client.frame(); await settle();
  for (let i = 0; i < 4; i++) { r.client.frame(); await settle(); }
  assert.ok(r.calls.filter(c => c[0] === 'cells' && c[1] === 22).length <= 2, 'at most once more after its summary');
  assert.ok(r.calls.filter(c => c[0] === 'summary' && c[1] === 22).length <= 1);
});

test("while the run rests, its newest result shows its own iteration's numbers (H2): the panel, the chip, the line; the smoothed one again after Resume; a saved one unchanged", async () => {
  const own = { per100: 8.79, shown: 9, change: { matched: -0.4, threshold: 1, model_threshold: 2.3, meaningful: false, direction: 'same' } };
  const r = rig(); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18, { per100: 11.2, shown: 11, own, change: { matched: -1.5, threshold: 1, meaningful: true, direction: 'fewer', held: true, direction_now: 'same' } }));
  r.client.frame();
  assert.deepEqual([r.view().summary.per100, r.view().summary.shown, r.view().summary.single], [11.2, 11, false], 'live: smoothed');
  assert.equal(r.view().ownShown, false); const smoothedVersion = r.view().lineVersion;
  // Paused (proseg acknowledged the pause) on the newest frame: the newest result held shows its own numbers, its
  // change unheld.
  Object.assign(r.page, { seq: 39, live: false, run: { ...r.page.run, state: 'paused', ack: { state: 'paused' } } }); r.client.frame();
  const v = r.view();
  assert.deepEqual([v.status, v.summary.per100, v.summary.shown, v.summary.single, v.summary.change], ['paused', 8.79, 9, true, own.change]);
  assert.equal(v.ownShown, true);
  assert.deepEqual(v.line.find(p => p.seq === 18), { ...r.client.points.get(18), per100: 8.79, shown: 9, single: true }, 'its point of the line too');
  assert.notEqual(v.lineVersion, smoothedVersion, 'the line is drawn again'); assert.equal(r.view().line, v.line);
  r.client.frame(); assert.equal(r.view().line, v.line, 'the same line while nothing changed');
  assert.deepEqual([r.client.summaries.get(18).per100, r.client.summaries.get(18).single], [11.2, false], 'the stored summary unchanged');
  assert.equal(r.client.points.get(18).per100, 11.2);
  // Pausing (no ack yet) is not a rest; finishing is.
  r.page.run = { ...r.page.run, state: 'paused', ack: { state: 'running' } }; r.client.frame(); assert.equal(r.view().summary.per100, 11.2);
  r.page.run = { ...r.page.run, state: 'finishing', ack: { state: 'paused' } }; r.client.frame(); assert.equal(r.view().summary.per100, 8.79);
  // Resumed: the smoothed result again.
  Object.assign(r.page, { seq: 20, live: true, run: { ...r.page.run, state: 'running', ack: { state: 'running' } } }); r.client.frame();
  assert.deepEqual([r.view().summary.per100, r.view().summary.single, r.view().summary.change.held], [11.2, false, true]);
  assert.equal(r.view().line.find(p => p.seq === 18).per100, 11.2); assert.equal(r.view().lineVersion, smoothedVersion);
  // A saved result (the engine's count "sampled": single, its own numbers already) is shown as it is.
  r.client.event(ready(19, { count: 'sampled', single: true, per100: 12.4, shown: 12, own: { per100: 12.4, shown: 12, change: own.change } }));
  Object.assign(r.page, { seq: 39, live: false, run: { ...r.page.run, state: 'paused', ack: { state: 'paused' } } }); r.client.frame();
  assert.deepEqual([r.view().summary.seq, r.view().summary.per100, r.view().summary.count, r.view().ownShown], [19, 12.4, 'sampled', false]);
  // Looking back at frame 18 while paused: not the rest's result, so as it was shown live.
  r.page.seq = 18; r.client.frame(); assert.deepEqual([r.view().summary.per100, r.view().summary.single], [11.2, false]);
});

test("a stopped run whose newest frame was never counted shows its newest result's own numbers, and that result's iteration (H2)", async () => {
  // Proseg stopped on an uncounted frame (39): the rest's result is the newest analysed counted frame's (18), never the
  // newest frame's (none).
  const own = { per100: 8.79, shown: 9, change: null };
  const r = rig(); r.page.seq = 20; r.client.reset(r.page.run); await settle();
  r.client.event(ready(18, { own })); r.client.frame();
  Object.assign(r.page, { seq: 39, live: false, run: { ...r.page.run, state: 'stopped' } }); r.client.frame();
  const v = r.view();
  assert.deepEqual([v.status, v.summary.seq, v.summary.iter, v.summary.per100, v.summary.shown, v.summary.single], ['finished', 18, 180, 8.79, 9, true]);
  for (const state of ['done', 'failed']) { r.page.run = { ...r.page.run, state }; r.client.frame(); assert.equal(r.view().summary.per100, 8.79, state); }
  // A summary without `own` (an older engine's) is shown as it is.
  const old = rig(); old.page.seq = 20; old.client.reset(old.page.run); await settle();
  old.client.event(ready(18)); Object.assign(old.page, { seq: 39, live: false, run: { ...old.page.run, state: 'stopped' } }); old.client.frame();
  assert.deepEqual([old.view().summary.per100, old.view().summary.single, old.view().ownShown], [11, false, false]);
});
