import test from 'node:test';
import assert from 'node:assert/strict';
import { Timeline, reduceFrames, columnsOf, nextDomain, REDRAW_MS, laneColumns, laneRange, LANE_H, LANE_MIN_HEIGHT, PHASES, scheduleEnd, withContinued,
  scheduledTemperature, scheduleTemperature, temperatureSchedule, axisTicks } from '../src/timeline.js';
import { QC_TEXT } from '../src/qc-copy.mjs';
import { fakeLoop, fakeElement } from './fake-loop.mjs';

const frames = (n, { from = 0, gapAfter = Infinity } = {}) => Array.from({ length: n }, (_, i) => {
  const seq = from + i, iter = seq <= gapAfter ? seq : gapAfter + (seq - gapAfter) * 5;
  return { seq, iter, phase: seq < 40 ? 'burnin' : 'sampling', changed: (seq * 37) % 101, loglik: seq % 7 ? -1e6 + seq : null,
    temperature: 1, live: { temperature: seq < 30 ? 1 : 0.5, compactness: seq < 60 ? 0.03 : 0.05 } };
});

// The reduction the timeline did per draw before (pixel columns in a Map; a JSON key per frame for knob changes).
function reference(list, x) {
  const columns = new Map(); let maxChanged = 1, lo = Infinity, hi = -Infinity, previousLive = ''; const ticks = new Set();
  for (let i = 0; i < list.length; i++) {
    const f = list[i], cx = Math.floor(x(i)), changed = Math.max(0, f.changed || 0);
    let c = columns.get(cx);
    if (!c) { c = { x: cx, min: changed, max: changed, lo: Infinity, hi: -Infinity }; columns.set(cx, c); }
    c.min = Math.min(c.min, changed); c.max = Math.max(c.max, changed); maxChanged = Math.max(maxChanged, changed);
    if (Number.isFinite(f.loglik)) { c.lo = Math.min(c.lo, f.loglik); c.hi = Math.max(c.hi, f.loglik); lo = Math.min(lo, f.loglik); hi = Math.max(hi, f.loglik); }
    if (f.live) { const live = JSON.stringify(Object.entries(f.live).sort(([a], [b]) => a.localeCompare(b))); if (i && live !== previousLive) ticks.add(cx); previousLive = live; }
  }
  return { columns: [...columns.values()], maxChanged, lo, hi, ticks: [...ticks] };
}

test('incremental frame reduction matches a full pass: columns, ranges, knob-change ticks and thinning', () => {
  const all = frames(150, { gapAfter: 120 }), list = [];
  let r = null;
  for (const f of all) { list.push(f); r = reduceFrames(r, list); }
  assert.equal(r.n, 150); assert.equal(r.thinned, true);
  assert.equal(reduceFrames(null, all.slice(0, 120)).thinned, false);
  for (const [left, scale] of [[12, 3.7], [12, 0.31], [12, 0]]) {
    const { columns, ticks } = columnsOf(r, left, scale), expected = reference(all, i => left + i * scale);
    assert.deepEqual(columns, expected.columns); assert.deepEqual(ticks, expected.ticks);
    assert.equal(r.maxChanged, expected.maxChanged); assert.equal(r.lo, expected.lo); assert.equal(r.hi, expected.hi);
  }
  assert.notEqual(reduceFrames(r, [...all]), r, 'a different frame list starts over');
});

/** A timeline drawing into a recording context: the texts it writes ([text, x, y]). */
function recordingTimeline() {
  const loop = fakeLoop(), texts = [];
  const g = new Proxy({}, { get: (t, k) => k in t ? t[k] : k === 'measureText' ? () => ({ width: 50 }) : k === 'createLinearGradient' ? () => ({ addColorStop() {} })
    : k === 'fillText' ? (text, x, y) => texts.push([text, x, y]) : () => {}, set: (t, k, v) => { t[k] = v; return true; } });
  const canvas = fakeElement({ clientWidth: 1000, clientHeight: 116, getContext: () => g });
  const timeline = loop.with(() => new Timeline(canvas, { onScrub() {} }));
  return { loop, texts, timeline, bands: () => texts.filter(t => t[1] > 0 && /^[A-Z][A-Z -]+$/.test(t[0])).map(t => t[0]) };
}
const at = (seq, phase, iter = seq) => ({ seq, iter, phase, changed: 1, loglik: -1, temperature: 1 });

test('proseg\'s standard run as its phases, then an open "continued" phase once the run samples on past its end', () => {
  const h = recordingTimeline();
  const plan = [{ phase: 'init', frames: 1 }, { phase: 'burnin', frames: 20 }, { phase: 'sampling', frames: 20 }, { phase: 'hillclimb', frames: 10 }];
  const list = [at(0, 'init'), ...Array.from({ length: 50 }, (_, i) => at(i + 1, i < 20 ? 'burnin' : i < 40 ? 'sampling' : 'hillclimb'))];
  h.loop.with(() => { h.timeline.setPlan(plan); h.timeline.setFrames(list); h.timeline.setCurrent(50, true); });
  h.loop.run(100);
  assert.equal(h.timeline.openEnded, false); assert.equal(h.timeline.total, 51, 'the standard run fills the axis at its end');
  assert.deepEqual(h.bands(), ['BURN-IN', 'SAMPLING', 'HILL-CLIMB']);
  // Resume past the end: frames of phase "continued" add an open phase; the axis grows with them.
  h.texts.length = 0;
  const more = [...list, ...Array.from({ length: 10 }, (_, i) => at(51 + i, 'continued'))];
  h.loop.with(() => { h.timeline.setFrames(more); h.timeline.setCurrent(60, true); });
  h.loop.run(400);
  assert.equal(h.timeline.openEnded, true); assert.ok(h.timeline.total > 61, 'room ahead on an open-ended axis');
  assert.deepEqual(h.bands(), ['BURN-IN', 'SAMPLING', 'HILL-CLIMB', 'CONTINUED']);
  assert.ok(!h.texts.some(t => /until/i.test(t[0])), 'no "until you stop"');
  assert.deepEqual(h.timeline.basePlan, plan, 'proseg\'s own plan is kept as it came');
  // A new run's plan starts without it again.
  h.loop.with(() => { h.timeline.setFrames([]); h.timeline.setPlan(plan); });
  assert.equal(h.timeline.openEnded, false);
});

test('a run made open-ended before 2026-10-18 keeps its open phase, labelled by its name alone', () => {
  const h = recordingTimeline();
  const plan = [{ phase: 'init', frames: 1 }, { phase: 'burnin', frames: 20 }, { phase: 'sampling', frames: 10 }, { phase: 'recorded', frames: null }];
  const list = [at(0, 'init'), ...Array.from({ length: 45 }, (_, i) => at(i + 1, i < 20 ? 'burnin' : i < 30 ? 'sampling' : 'recorded'))];
  h.loop.with(() => { h.timeline.setPlan(plan); h.timeline.setFrames(list); h.timeline.setCurrent(45, true); });
  h.loop.run(400);
  assert.deepEqual(h.bands(), ['BURN-IN', 'SAMPLING', 'RECORDED']);
  assert.ok(!h.texts.some(t => /until/i.test(t[0])));
});

test('the hill-climb\'s own cooling (and the return to 1 past it) is never marked or listed as a knob change; a change of the user\'s is', () => {
  const cooling = t => ({ temperature: t, 'cell-compactness': 0.04 });
  const list = [
    { ...at(0, 'recorded'), live: cooling(1) }, { ...at(1, 'hillclimb'), live: cooling(1) },
    { ...at(2, 'hillclimb'), temperature: 0.98, live: cooling(0.98) }, { ...at(3, 'hillclimb'), temperature: 0.95, live: cooling(0.95) },
    { ...at(4, 'hillclimb'), temperature: 0.93, live: { ...cooling(0.93), 'cell-compactness': 0.02 } },   // the user's change
    { ...at(5, 'continued'), live: { ...cooling(1), 'cell-compactness': 0.02 } },                         // back to 1 by itself
    { ...at(6, 'continued'), temperature: 0.5, live: { ...cooling(0.5), 'cell-compactness': 0.02 } },     // the user's temperature
  ];
  const r = reduceFrames(null, list);
  assert.deepEqual(r.tick, [false, false, false, false, true, false, true]);
  const h = recordingTimeline();
  h.loop.with(() => { h.timeline.setPlan([{ phase: 'recorded', frames: 1 }, { phase: 'hillclimb', frames: 4 }]); h.timeline.setFrames(list); h.timeline.setCurrent(6, true); });
  h.loop.run(100);
  const hover = seq => { h.loop.with(() => h.timeline.canvas.listeners.pointermove({ offsetX: h.timeline.x(seq), clientX: 0, clientY: 900 })); h.loop.run(h.loop.now + 50); return h.timeline.readout.textContent; };
  assert.doesNotMatch(hover(3), /→/, 'cooling alone: no change line (its temperature line says it)');
  const at4 = hover(4);
  assert.match(at4, /^Cell compactness 0\.04 → 0\.02/m); assert.doesNotMatch(at4, /^Temperature .*→/m);
  assert.doesNotMatch(hover(5), /→/);
  assert.match(hover(6), /→ 0\.5/, 'a temperature the user set is a change');
});

test('proseg\'s own temperature schedule decides, not the phase: a Temperature set in the hill-climb or at the hold is the user\'s change', () => {
  // burn-in 200 and sampling 200 (proseg's defaults): the hill-climb cools 0.01^(1/200) per iteration from iteration 402;
  // past the end the schedule gives 1 again. The help's own advice ("lower Temperature first", then Resume) used to be
  // hidden from the timeline and the log as "proseg's own".
  const run = { info: { schedule: { burnin: 200, samples: 200, recorded: 100, hillclimb: 50 } }, options: {} };
  const sched = temperatureSchedule(run), cool = i => Math.exp(Math.log(0.01) / 200 * (i - 401));
  assert.deepEqual(sched, { burnin: 200, samples: 200 }); assert.equal(temperatureSchedule({}), null);
  assert.ok(Math.abs(scheduleTemperature({ phase: 'hillclimb', iter: 450 }, sched) - 0.323594) < 1e-6);
  assert.equal(scheduleTemperature({ phase: 'continued', iter: 451 }, sched), 1);
  const frame = (iter, phase, t) => ({ iter, phase, temperature: t, live: { temperature: t, 'cell-compactness': 0.04 } });
  const phase = i => i > 450 ? 'continued' : i > 400 ? 'hillclimb' : 'recorded';
  const held = [449, 450, 451, 452].map(i => frame(i, phase(i), i <= 450 ? cool(i) : 0.1));   // 0.1 set at the hold
  assert.deepEqual(reduceFrames(null, held, sched).tick, [false, false, true, false]);
  assert.deepEqual(reduceFrames(null, held).tick, [false, false, false, false], 'the old rule (no schedule) hid it');
  const plain = [449, 450, 451].map(i => frame(i, phase(i), i <= 450 ? cool(i) : 1));       // nothing set: back to 1 by itself
  assert.deepEqual(reduceFrames(null, plain, sched).tick, [false, false, false]);
  const during = [419, 420, 421, 422].map(i => frame(i, phase(i), i <= 420 ? cool(i) : 0.05)); // set during the hill-climb
  assert.deepEqual(reduceFrames(null, during, sched).tick, [false, false, true, false]);
  assert.equal(scheduledTemperature(during[2], during[1], sched), false); assert.equal(scheduledTemperature(during[1], during[0], sched), true);
  // A temperature set at the start (Advanced options) holds for every iteration: no mark.
  const fixed = temperatureSchedule({ ...run, options: { temperature: 0.5 } });
  assert.deepEqual(reduceFrames(null, [frame(0, 'init', 1), frame(1, 'burnin', 0.5), frame(2, 'burnin', 0.5)], fixed).tick, [false, false, false]);
  // The timeline takes the run with its plan; its hover readout lists the user's temperature at the hold.
  const h = recordingTimeline();
  h.loop.with(() => { h.timeline.setPlan([{ phase: 'recorded', frames: 1 }, { phase: 'hillclimb', frames: 2 }], run); h.timeline.setFrames(held); h.timeline.setCurrent(3, true); });
  h.loop.run(100);
  h.loop.with(() => h.timeline.canvas.listeners.pointermove({ offsetX: h.timeline.x(2), clientX: 0, clientY: 900 })); h.loop.run(h.loop.now + 50);
  assert.match(h.timeline.readout.textContent, /→ 0\.1/);
});

test('axis labels sit at round iterations, so they and proseg\'s phases agree', () => {
  // The finer-voxel snapshot (a second one at iteration 200) shifted every later label: 219, 239 ... 299, 399, beside
  // bands that start at 301 and 401.
  const plan = [0, ...Array.from({ length: 200 }, (_, i) => i + 1), 200, ...Array.from({ length: 250 }, (_, i) => 201 + i)];
  const list = plan.map((iter, seq) => ({ seq, iter }));
  const ticks = axisTicks(list, list.length, 20);
  assert.deepEqual(ticks.slice(9, 13).map(t => t.label), [180, 200, 220, 240]);
  assert.deepEqual(ticks.filter(t => [300, 400].includes(t.label)).map(t => [t.label, list[t.seq].iter]), [[300, 300], [400, 400]]);
  assert.equal(ticks.find(t => t.label === 220).seq, 221, 'at the first snapshot that reached it');
  // Ahead of the newest snapshot: marks without labels, one snapshot per iteration.
  const early = axisTicks(list.slice(0, 50), 452, 100);
  assert.deepEqual(early.map(t => [t.seq, t.label]), [[0, 0], [100, null], [200, null], [300, null], [400, null]]);
  // A thinned run keeps evenly spaced snapshots labelled with their own iterations.
  assert.deepEqual(axisTicks([{ iter: 0 }, { iter: 5 }, { iter: 10 }, { iter: 15 }], 4, 2, true), [{ seq: 0, label: 0 }, { seq: 2, label: 10 }]);
});

test('the schedule\'s end and the continued phase as plain data', () => {
  const plan = [{ phase: 'init', frames: 1 }, { phase: 'burnin', frames: 200 }, { phase: 'upscale', frames: 1 }, { phase: 'sampling', frames: 100 },
    { phase: 'recorded', frames: 100 }, { phase: 'hillclimb', frames: 50 }];
  assert.equal(scheduleEnd({ schedule: { burnin: 200, samples: 200, recorded: 100, hillclimb: 50 }, plan }), 450, 'proseg\'s defaults');
  assert.equal(scheduleEnd({ schedule: { burnin: 100, samples: 60, hillclimb: 0 }, plan: [...plan.slice(0, 4), { phase: 'recorded', frames: null }] }), null);
  assert.equal(scheduleEnd({ plan }), null); assert.equal(scheduleEnd(undefined), null);
  assert.equal(withContinued(plan, [at(0, 'init'), at(1, 'hillclimb')]), plan, 'unchanged within the standard run');
  assert.deepEqual(withContinued(plan, [at(451, 'continued')]).at(-1), { phase: 'continued', frames: null });
  assert.equal(withContinued(plan, []), plan); assert.equal(withContinued(plan, undefined), plan);
  const open = [{ phase: 'recorded', frames: null }];
  assert.equal(withContinued(open, [at(9, 'continued')]), open, 'an open plan has no end to go past');
  assert.ok(PHASES.continued.label && PHASES.continued.help && !/until/i.test(PHASES.continued.help));
});

test('the open-ended axis snaps when one iteration arrives and glides only for larger jumps', () => {
  assert.equal(nextDomain(0, 500, 16), 500);
  assert.equal(nextDomain(500, 501.1, 100), 501.1);
  const glide = nextDomain(500, 900, 100);
  assert.ok(glide > 500 && glide < 900);
  assert.ok(nextDomain(glide, 900, 250) > glide);
});

function harness() {
  const loop = fakeLoop(), draws = [];
  const canvas = fakeElement({ clientWidth: 1000, clientHeight: 116 });
  const timeline = loop.with(() => new Timeline(canvas, { onScrub() {} }));
  const draw = timeline.draw.bind(timeline);
  timeline.draw = () => { draws.push(loop.now); draw(); };
  const list = [];
  return { loop, draws, canvas, timeline, list, add: (t, current = true) => loop.at(t, () => {
    list.push(...frames(1, { from: list.length })); timeline.setFrames(list); if (current) timeline.setCurrent(list.length - 1);
  }) };
}

test('following a live run redraws the timeline at ~10 fps at most, and nothing runs once it is idle', () => {
  const h = harness();
  h.loop.run(500);
  h.draws.length = 0;
  h.loop.with(() => h.timeline.setPlan([{ phase: 'burnin', frames: 40 }, { phase: 'sampling', frames: null }]));
  for (let t = 1000; t < 3000; t += 33) h.add(t);   // 30 iterations per second
  h.loop.run(5000);
  const live = h.draws.filter(t => t > 1000);
  for (let i = 1; i < live.length; i++) assert.ok(live[i] - live[i - 1] >= REDRAW_MS - 1e-6, `gap ${live[i] - live[i - 1]} ms`);
  assert.ok(live.length <= 22, `${live.length} draws for 2 s of frames`);
  assert.equal(h.timeline.current, h.list.length - 1);
  assert.ok(h.loop.idle(), 'no animation loop once everything is drawn');
});

test('pointer interaction and user seeks draw on the next frame; a hidden page draws nothing until shown', () => {
  const h = harness();
  h.loop.with(() => h.timeline.setPlan([{ phase: 'burnin', frames: 200 }]));
  for (let k = 0; k < 50; k++) h.add(100 + k, false);
  h.loop.run(1000);
  h.draws.length = 0;
  h.loop.at(1000, () => h.timeline.setCurrent(10));            // a live playhead update...
  h.loop.at(1005, () => h.canvas.listeners.pointermove({ offsetX: 300, clientX: 300, clientY: 900 }));   // ...then the pointer
  h.loop.run(1100);
  assert.equal(h.draws.length, 1); assert.ok(h.draws[0] < 1020, 'hover draws on the next animation frame');
  h.loop.at(1200, () => h.timeline.setCurrent(20, true));       // a scrub or step
  h.loop.run(1300);
  assert.equal(h.draws.length, 2); assert.ok(h.draws[1] < 1220);
  h.loop.setHidden(true);
  h.loop.at(1400, () => { h.timeline.setCurrent(30, true); h.timeline.setFrames(h.list); });
  h.loop.run(3000);
  assert.equal(h.draws.length, 2); assert.ok(h.loop.idle());
  h.loop.setHidden(false);
  h.loop.run(3100);
  assert.equal(h.draws.length, 3);
  // Covered by Compare's full-screen view (main.js): the same, until it goes.
  h.loop.with(() => h.timeline.setCovered(true));
  h.loop.at(3200, () => { h.timeline.setCurrent(40, true); h.timeline.setFrames(h.list); });
  h.loop.run(5000);
  assert.equal(h.draws.length, 3); assert.ok(h.loop.idle());
  h.loop.with(() => h.timeline.setCovered(false));
  h.loop.run(5100);
  assert.equal(h.draws.length, 4);
});

test('the hover readout is measured once per hovered iteration, not on every live redraw', () => {
  const h = harness();
  let measured = 0;
  const readout = h.timeline.readout;
  Object.defineProperty(readout, 'offsetWidth', { get: () => { measured++; return 120; } });
  h.loop.with(() => h.timeline.setPlan([{ phase: 'burnin', frames: 400 }]));
  for (let k = 0; k < 20; k++) h.add(10 + k, false);
  h.loop.at(500, () => h.canvas.listeners.pointermove({ offsetX: 20, clientX: 20, clientY: 900 }));
  for (let k = 0; k < 20; k++) h.add(600 + 150 * k);
  h.loop.run(4000);
  assert.equal(readout.hidden, false);
  assert.equal(measured, 1);
  h.loop.at(4100, () => h.canvas.listeners.pointerleave());
  h.loop.run(4200);
  assert.equal(readout.hidden, true);
});

test('the mixing lane: columns per pixel with their range, its scale fitted to the formed line and the start', () => {
  const points = [{ seq: 0, per100: 2 }, { seq: 1, per100: 4 }, { seq: 2, per100: null }, { seq: 3, per100: 12.5 }, { seq: 4, per100: 11 }];
  assert.deepEqual(laneColumns(points, 10, 0.5), [{ x: 10, min: 2, max: 4, last: 4 }, { x: 11, min: 12.5, max: 12.5, last: 12.5 }, { x: 12, min: 11, max: 11, last: 11 }]);
  assert.deepEqual(laneColumns(points, 10, 0.3), [{ x: 10, min: 2, max: 12.5, last: 12.5 }, { x: 11, min: 11, max: 11, last: 11 }], 'one column keeps its range and its last value');
  assert.deepEqual(laneColumns(points, 0, 10).map(c => c.x), [0, 10, 30, 40], 'a missing value is a gap, not a zero');
  // The scale fits the line once cells have formed and the start, 2 points around them: a change of a point shows.
  const run = [...Array.from({ length: 10 }, (_, i) => ({ seq: i, iter: i * 10, per100: 6 + i * .7 })),
    ...Array.from({ length: 10 }, (_, i) => ({ seq: 10 + i, iter: 100 + i * 10, per100: 11 + (i % 2) * .5 }))];
  const cols = laneColumns(run, 0, 10, 90);
  assert.equal(cols.filter(c => c.forming).length, 10, 'the forming stretch is marked (faint, outside the scale)');
  assert.deepEqual(laneRange(cols, 12), [9, 14]); assert.deepEqual(laneRange(laneColumns(run.slice(0, 5), 0, 10, 90), null), [4, 11]);
  assert.deepEqual(laneRange([], null), [0, 5]); assert.deepEqual(laneRange([{ min: 50, max: 50 }], 50), [48, 52]);
});

test('the mixing lane is drawn inside the timeline\'s own height (the stage never changes size for it); its hover line says the number', () => {
  const loop = fakeLoop(), texts = [], fills = [];
  const g = new Proxy({}, { get: (t, k) => k in t ? t[k] : k === 'measureText' ? () => ({ width: 50 }) : k === 'createLinearGradient' ? () => ({ addColorStop() {} })
    : k === 'fillText' ? (text, x, y) => texts.push([text, x, y]) : k === 'fillRect' ? (...a) => fills.push(a) : () => {}, set: (t, k, v) => { t[k] = v; return true; } });
  const canvas = fakeElement({ clientWidth: 1000, clientHeight: 116, getContext: () => g });
  const timeline = loop.with(() => new Timeline(canvas, { onScrub() {} }));
  const list = frames(60);
  loop.with(() => { timeline.setPlan([{ phase: 'burnin', frames: 40 }, { phase: 'sampling', frames: 20 }]); timeline.setFrames(list); timeline.setCurrent(59, true); });
  loop.run(200);
  const axis = texts.filter(t => /^\d+$/.test(t[0])).map(t => t[2]);
  assert.ok(axis.every(y => y === 113), 'iteration labels at the bottom without the lane');
  const lane = { points: list.map(f => ({ seq: f.seq, per100: 5 + f.seq / 10, shown: Math.round(5 + f.seq / 10) })), version: 1, start: 6 };
  texts.length = 0; loop.with(() => timeline.setQc(lane)); loop.run(400);
  assert.ok(LANE_MIN_HEIGHT <= 116, 'the app\'s 116 px timeline has room for it');
  assert.ok(texts.some(t => t[0] === QC_TEXT.timelineLegend), 'the lane\'s legend');
  assert.ok(texts.some(t => t[0] === 'Iteration 0: 6'), "the start named after what it is: proseg's count at iteration 0 (H3)");
  const value = texts.find(t => t[0] === '11'), legend = texts.find(t => t[0] === QC_TEXT.timelineLegend);
  assert.ok(value && value[2] > 113 - LANE_H, 'the newest value at the line\'s end, in the lane');
  assert.ok(legend[1] < value[1] - 50 || legend[1] > value[1], 'the legend is not over the line\'s newest stretch');
  assert.deepEqual(timeline.qcTexts().sort(), ['Iteration 0: 6', '11', QC_TEXT.timelineLegend].sort(), 'its words, for the harness');
  assert.ok(texts.filter(t => /^\d+$/.test(t[0]) && t !== value).every(t => t[2] === 113 - LANE_H), 'the chart above it is LANE_H px shorter: the canvas keeps its size');
  texts.length = 0; timeline.width = 1000; loop.with(() => { timeline.height = LANE_MIN_HEIGHT - 1; timeline.invalidate(true); }); loop.run(600);
  assert.ok(!texts.some(t => t[0] === QC_TEXT.timelineLegend), 'a smaller canvas has no room: no squeezed lane');
  texts.length = 0; loop.with(() => { timeline.height = 116; timeline.invalidate(true); }); loop.run(700);
  assert.ok(texts.some(t => t[0] === QC_TEXT.timelineLegend));
  const draws = timeline.drawCount;
  loop.with(() => timeline.setQc({ ...lane })); loop.run(800);
  assert.equal(timeline.drawCount, draws, 'the same lane again draws nothing');
  loop.with(() => canvas.listeners.pointermove({ offsetX: 990, clientX: 990, clientY: 900 })); loop.run(900);
  assert.match(timeline.readout.textContent, /\n11 in 100 cells look mixed$/, 'an engine that says neither: plain');
  // H2: a live result's number is smoothed, a saved iteration's (count "sampled") as saved, an own one plain.
  const hoverWith = (point, at) => {
    loop.with(() => timeline.setQc({ ...lane, version: at, points: lane.points.map(p => p.seq === 59 ? { ...p, ...point } : p) })); loop.run(at);
    loop.with(() => canvas.listeners.pointermove({ offsetX: 990, clientX: 990, clientY: 900 })); loop.run(at + 200);
    return timeline.readout.textContent.split('\n').at(-1);
  };
  assert.equal(hoverWith({ single: false, count: 'averaged' }, 1000), '11 in 100 cells look mixed, smoothed');
  assert.equal(hoverWith({ single: true, count: 'sampled' }, 1400), '11 in 100 cells look mixed, as saved');
  assert.equal(hoverWith({ single: true, count: 'averaged' }, 1800), '11 in 100 cells look mixed');
  loop.with(() => timeline.setQc(null)); loop.run(2300);
  assert.doesNotMatch(timeline.readout.textContent, /mixed/, 'the check off: no lane, no line in the readout');
  assert.ok(loop.idle());
});

test('the hover readout names a knob change by a vetted name or its plain change, with its values, never the raw flag nor a banned word', () => {
  const loop = fakeLoop(), canvas = fakeElement({ clientWidth: 1000, clientHeight: 116 });
  const labels = { 'cell-compactness': 'Cell compactness (lower = rounder cells)', 'prior-weight': 'Prior weight (higher = the starting cells hold their transcripts harder)',
    'no-diffusion': 'Switch off the transcript drift model (no diffusion)' };
  const timeline = loop.with(() => new Timeline(canvas, { onScrub() {}, knobLabel: flag => labels[flag] ?? null }));
  const list = Array.from({ length: 40 }, (_, seq) => ({ seq, iter: seq, phase: 'sampling', changed: 1, loglik: -1, temperature: 1,
    // proseg reports knobs as 32-bit floats; 'new-knob' has no label at all.
    live: { 'cell-compactness': seq < 20 ? 0.03999999910593033 : 0.019999999552965164, 'prior-weight': seq < 30 ? 1 : 4, 'no-diffusion': seq >= 30,
      temperature: 1, ...(seq >= 30 ? { 'new-knob': 2 } : {}) } }));
  loop.with(() => { timeline.setPlan([{ phase: 'sampling', frames: 40 }]); timeline.setFrames(list); timeline.setCurrent(39, true); });
  loop.run(100);
  const hover = seq => { loop.with(() => canvas.listeners.pointermove({ offsetX: timeline.x(seq), clientX: 0, clientY: 900 })); loop.run(loop.now + 50); return timeline.readout.textContent; };
  const at20 = hover(20);
  assert.match(at20, /^Cell compactness 0\.04 → 0\.02 \(tighter shapes\)$/m);
  assert.doesNotMatch(at20, /cell-compactness|0\.0399|0\.0199/, 'no raw flag, no float noise');
  assert.doesNotMatch(hover(21), /Cell compactness/, 'a frame without a change lists none');
  const at30 = hover(30);
  // Prior weight's Live knobs name holds a banned word (it may stand only in a Setting line): its plain change; a switch
  // by what it does; a knob absent before reads "—", never "automatic".
  assert.match(at30, /^firmer hold: 1 → 4$/m);
  assert.match(at30, /^drift off$/m);
  assert.match(at30, /^Another setting changed: — → 2$/m);
  assert.doesNotMatch(at30, /prior-weight|no-diffusion|new-knob|Prior weight|Switch off/);
  // Without the option schema (no labels yet): the plain change alone, still never the flag.
  const bare = loop.with(() => new Timeline(fakeElement({ clientWidth: 1000, clientHeight: 116 }), { onScrub() {} }));
  bare.frames = list;
  assert.equal(bare.knobLabel('cell-compactness'), null);
});
