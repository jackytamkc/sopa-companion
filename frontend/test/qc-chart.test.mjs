import test from 'node:test';
import assert from 'node:assert/strict';
import { linearScale, niceTicks, shownInteger, seriesPoints, flatBand, trialBand, canPredict, TRIAL_BAND, layoutChart,
  layoutSparkline, layoutTimeline, hoverAt, renderChart, QC_COLORS } from '../src/qc-chart.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-8, `${a} != ${b}`);
const series = [0, 10, 20, 30].map((iter, i) => ({ seq: i + 1, iter, per100: i * 5, judged: 1000 + i, phase: i < 2 ? 'burn-in' : 'sampling' }));
function canvas() {
  const calls = [], context = new Proxy({}, { get: (o, name) => name in o ? o[name] : (...args) => {
    calls.push([name, ...args]); if (name === 'measureText') return { width: String(args[0]).length * 5 };
  } });
  return { width: 0, height: 0, style: {}, calls, getContext: () => context };
}

test('linear scales invert, degenerate scales stay finite, ticks are useful at all magnitudes', () => {
  const scale = linearScale([0, 100], [20, 280]);
  assert.equal(scale(50), 150); assert.equal(scale.invert(150), 50);
  assert.equal(linearScale([5, 5], [0, 10])(5), 5);
  assert.equal(linearScale([0, 10], [5, 5]).invert(5), 5);
  assert.deepEqual(niceTicks(0, 15, 3), [0, 5, 10, 15]);
  assert.deepEqual(niceTicks(0, 1, 5), [0, .2, .4, .6, .8, 1]);
  assert.deepEqual(niceTicks(7, 7), [7]); assert.deepEqual(niceTicks(NaN, 2), []);
});

test('display rounding crosses a boundary by .3 and does not flicker', () => {
  assert.equal(shownInteger(8.5), 9);
  let n = 8;
  for (const x of [8.49, 8.51, 8.79, 8.6]) { n = shownInteger(x, n); assert.equal(n, 8); }
  assert.equal(shownInteger(8.8, n), 9); assert.equal(shownInteger(8.21, 9), 9);
  assert.equal(shownInteger(8.2, 9), 8); assert.equal(shownInteger(12.6, 8), 12);
  assert.equal(shownInteger(99.9, 99), 100); assert.equal(shownInteger(-1, 1), 0);
  assert.equal(shownInteger(null, 8), null);
});

test('layout sorts copies, preserves gaps and uses the last duplicate iteration', () => {
  const data = [{ iter: 20, per100: 12 }, { iter: 10, per100: null }, { iter: 0, per100: 0 }, { iter: 20, per100: 13 }];
  const original = JSON.stringify(data), points = seriesPoints(data), layout = layoutChart({ series: data, width: 264 });
  assert.deepEqual(points.map(p => p.per100), [0, null, 13]);
  assert.equal(layout.points[1].y, null); assert.equal(layout.hoverPoints.length, 2);
  assert.equal(JSON.stringify(data), original); near(layout.points[0].x, layout.plot.left);
  near(layout.points.at(-1).x, layout.plot.right); assert.ok(layout.yDomain[1] >= 13);
  assert.ok(layout.points.every(p => p.y == null || p.y >= layout.plot.top && p.y <= layout.plot.bottom));
});

test('start reference includes zero, forming band ends at sampling, ticks carry plain labels', () => {
  const layout = layoutChart({ series, start: { per100: 10 }, changes: [{ iter: 25, key: 'tighterShapes' }] });
  // H3: named after what it is, proseg's count at iteration 0: never "start 10", nor the cells the run started from.
  assert.equal(layout.start.label, 'Iteration 0: 10', 'named after what it is, never "start 10"'); near(layout.start.y, layout.y(10));
  assert.equal(layoutChart({ series, start: 10, startName: 'the starting cells' }).start.label, 'Iteration 0: 10', 'whatever the cells were');
  near(layout.formingBand.x1, layout.x(20)); assert.equal(layout.formingBand.label, 'cells forming');
  assert.equal(layout.changes[0].label, 'tighter shapes'); near(layout.changes[0].x, layout.x(25));
  assert.equal(layoutChart({ series, start: 0 }).start.value, 0);
  assert.equal(layoutChart({ series, start: null }).start, null);
  assert.equal(layoutChart({ series, formingUntil: 200 }).formingBand.x1, layout.plot.right);
});

test('hover uses CSS pixels, handles gaps, ties and edges, and names the last change', () => {
  const layout = layoutChart({ series, changes: [{ iter: 25, key: 'tighterShapes', flag: 'cell-compactness', from: 0.04, value: 0.02 },
    { iter: 10, key: 'driftOn', flag: 'no-diffusion', from: true, value: false }] });
  assert.equal(hoverAt(layout, layout.x(15)).iter, 10);
  assert.equal(hoverAt(layout, layout.x(21)).iter, 20);
  assert.equal(hoverAt(layout, layout.x(30)).text, 'Iteration 30 · 15.0 in 100 cells look mixed (of 1,003 with 100 or more transcripts)'
    + ' · after Cell compactness 0.04 → 0.02 (tighter shapes), iteration 25', 'the Live knobs name with its values');
  assert.equal(hoverAt(layout, layout.x(20)).text, 'Iteration 20 · 10.0 in 100 cells look mixed (of 1,002 with 100 or more transcripts)'
    + ' · after drift on (on → off), iteration 10');
  assert.equal(hoverAt(layout, layout.plot.left - 1), null);
  assert.equal(hoverAt(layout, layout.x(15), { maxDistance: 1 }), null);
  assert.equal(hoverAt(layoutChart(), 30), null);
});

test("the hover says how a point's number was made: smoothed while live, as saved at a saved iteration, else plain (H2)", () => {
  // A live result (single false) is smoothed with the ones before it; a frame counted for a pause, the hold, a finish
  // or the end (count "sampled") is the count proseg saves there, its own; a result judged alone (or a resting run's
  // own point) is plain.
  const points = [{ seq: 1, iter: 100, per100: 11.24, judged: 900, single: false, count: 'averaged' },
    { seq: 2, iter: 200, per100: 12.5, judged: 910, single: true, count: 'sampled' },
    { seq: 3, iter: 300, per100: 13, judged: 920, single: true, count: 'averaged' }, { seq: 4, iter: 400, per100: 14, judged: 930 }];
  const layout = layoutChart({ series: points });
  assert.equal(hoverAt(layout, layout.x(100)).text, 'Iteration 100 · 11.2 in 100 cells look mixed, smoothed (of 900 with 100 or more transcripts)');
  assert.equal(hoverAt(layout, layout.x(200)).text, 'Iteration 200 · 12.5 in 100 cells look mixed, as saved (of 910 with 100 or more transcripts)');
  assert.equal(hoverAt(layout, layout.x(300)).text, 'Iteration 300 · 13.0 in 100 cells look mixed (of 920 with 100 or more transcripts)');
  assert.equal(hoverAt(layout, layout.x(400)).text, 'Iteration 400 · 14.0 in 100 cells look mixed (of 930 with 100 or more transcripts)', 'an engine that says neither: plain');
  for (const x of [100, 200, 300]) assert.deepEqual(bannedWords(hoverAt(layout, layout.x(x)).text), []);
});

test('the trial band is flat: the mean of the 50 iterations before the change, from 20 after cells formed, ± max(2 SD, 1)', () => {
  // A line that rises steadily: a fitted line would predict it on; the flat band does not (68 % false verdicts, measured).
  const data = Array.from({ length: 80 }, (_, iter) => ({ iter: 100 + iter, per100: 10 + iter / 20, phase: 'sampling' }));
  data.push({ iter: 150.5, per100: 90, single: true });   // a single snapshot never counts
  const band = flatBand(data, { start_iter: 170 }, { formedIter: 100 });
  const history = data.filter(p => !p.single && p.iter >= 120 && p.iter < 170).map(p => p.per100);
  const mean = history.reduce((a, b) => a + b, 0) / history.length;
  const sd = Math.sqrt(history.reduce((a, b) => a + (b - mean) ** 2, 0) / (history.length - 1));
  near(band.centre, mean); near(band.sd, sd); near(band.halfWidth, Math.max(2 * sd, 1));
  assert.equal(band.from, 170); assert.equal(band.to, 179, 'drawn up to the newest point');
  assert.deepEqual(band.points.map(p => p.iter), [170, 179]);
  assert.ok(band.points.every(p => Math.abs(p.low - (mean - band.halfWidth)) < 1e-9 && Math.abs(p.high - (mean + band.halfWidth)) < 1e-9), 'flat');
  // Only from 20 iterations after cells formed on: here the history starts at 150 (10 results would still do, 9 do not).
  const late = flatBand(data, { start_iter: 170 }, { formedIter: 130 });
  assert.equal(late.points.length, 2); near(late.centre, data.filter(p => !p.single && p.iter >= 150 && p.iter < 170).reduce((a, p) => a + p.per100, 0) / 20);
  assert.equal(flatBand(data, { start_iter: 170 }, { formedIter: 141 }), null, 'fewer than 10 results: no band');
  assert.equal(flatBand(data.slice(0, 5), { start_iter: 105 }), null);
  // A quiet line: the floor of 1 in 100; a long trial: at most 300 iterations of band.
  const quiet = Array.from({ length: 500 }, (_, iter) => ({ iter, per100: 12 }));
  const q = flatBand(quiet, { start_iter: 100 }, { formedIter: 0 });
  assert.equal(q.halfWidth, TRIAL_BAND.floor); assert.equal(q.to, 100 + TRIAL_BAND.horizon, 'verdicts only up to 300 iterations: so is the band');
  for (const trial of [{ start_iter: 100, forming: true }, { start_iter: 100, notes: ['mixed_up'] }, { start_iter: 100, rule: 'R3', notes: ['no_verdict'] },
    { start_iter: 100, outcome: 'kept' }, { start_iter: null }]) assert.equal(flatBand(quiet, trial), null, JSON.stringify(trial));
  assert.equal(canPredict({ start_iter: 3 }), true); assert.equal(canPredict(null), false);
  // Drift (R3) shows in proseg's count: it gets its band like any other change (an older engine's "no verdict" note: none).
  assert.equal(canPredict({ start_iter: 3, rule: 'R3', flag: 'no-diffusion' }), true);
  assert.deepEqual(flatBand(quiet, { start_iter: 100, rule: 'R3' }, { formedIter: 0 }), q);
});

test("the engine's band wins over the drawn one, in any of its forms; none while cells formed, mixed up or ended; drift's like any other", () => {
  const data = [10, 16, 10, 16].map((per100, iter) => ({ iter, per100 }));
  const supplied = [{ iter: 2, low: 8, high: 15 }, { iter: 4, low: 9, high: 16 }];
  const layout = layoutChart({ series: data, trial: { start_iter: 2, band: supplied } });
  assert.equal(layout.prediction, null); assert.equal(layout.band.length, 2);
  near(layout.band[1].yHigh, layout.y(16)); near(layout.band[0].yLow, layout.y(8));
  // engine/qc_rules.py Watch.band_points: two points of a flat band; or a flat object.
  assert.deepEqual(trialBand({ start_iter: 2, band: { centre: 11, half_width: 1.5 } }, 40).map(p => [p.iter, p.low, p.high]), [[2, 9.5, 12.5], [40, 9.5, 12.5]]);
  assert.deepEqual(trialBand({ start_iter: 2, band: { low: 9, high: 12, from_iter: 5, to_iter: 9 } }).map(p => p.iter), [5, 9]);
  assert.deepEqual(trialBand({ start_iter: 2, band: { low: 9, high: 12 } }, 900).map(p => p.iter), [2, 302], 'at most 300 iterations on');
  assert.equal(trialBand({ start_iter: 2, band: { low: 12, high: 9 } }), null); assert.equal(trialBand({ start_iter: 2 }), null);
  for (const trial of [{ forming: true, band: supplied }, { notes: ['forming'], band: supplied }, { outcome: 'kept', band: supplied },
    { rule: 'R3', band: supplied, notes: ['no_verdict'] }, { mixed_up: true, band: supplied }]) assert.equal(layoutChart({ series: data, trial }).band.length, 0, JSON.stringify(trial));
  // Drift (R3) is judged live on proseg's count: the engine's band is drawn for it as for any other trial.
  const drift = layoutChart({ series: data, trial: { start_iter: 2, rule: 'R3', flag: 'no-diffusion', band: supplied, notes: [] } });
  assert.equal(drift.band.length, 2); assert.equal(drift.prediction, null); near(drift.band[0].yLow, drift.y(8));
  const flat = layoutChart({ series: data, trial: { start_iter: 2, rule: 'R3', band: { centre: 10.2, half_width: 1.1 }, notes: [] } }).band;
  assert.deepEqual(flat.map(p => p.iter), [2, 3]); for (const p of flat) { near(p.low, 9.1); near(p.high, 11.3); }
  // Without the engine's, the flat band drawn from the line itself (the chart's own forming stretch decides where it starts).
  const line = Array.from({ length: 40 }, (_, i) => ({ iter: 100 + i, per100: 11 + (i % 2) * .4 }));
  const drawn = layoutChart({ series: line, trial: { start_iter: 130 }, formingUntil: 95 });
  assert.ok(drawn.prediction && drawn.band.length === 2); near(drawn.band[0].low, drawn.prediction.centre - drawn.prediction.halfWidth);
  assert.ok(drawn.yDomain[0] <= drawn.band[0].low && drawn.yDomain[1] >= drawn.band[0].high, 'the band is inside the scale');
  // An engine trial says when it has no band (null: too little history, mixed up): none is drawn in its place, for
  // drift neither.
  for (const trial of [{ start_iter: 130, band: null, notes: ['no_history'] }, { start_iter: 130, band: null, notes: [] },
    { start_iter: 130, band: null, notes: ['no_verdict'] }, { start_iter: 130, band: null, notes: ['last_stage'] },
    { start_iter: 130, rule: 'R3', flag: 'no-diffusion', band: null, notes: [] }]) {
    const layout = layoutChart({ series: line, trial, formingUntil: 95 });
    assert.equal(layout.band.length, 0, JSON.stringify(trial)); assert.equal(layout.prediction, null);
  }
  assert.equal(canPredict({ start_iter: 130, notes: ['no_history'] }), false);
});

test('sparkline is exactly 48 by 14; timeline lane has its own scale', () => {
  const spark = layoutSparkline(series, { width: 500, start: 10 });
  assert.equal(spark.width, 48); assert.equal(spark.height, 14); assert.equal(spark.xTicks.length, 0);
  assert.ok(spark.points.every(p => Number.isFinite(p.x) && Number.isFinite(p.y)));
  for (const value of [0, 100]) assert.ok(layoutSparkline([{ iter: 0, per100: value }]).points.every(p => Number.isFinite(p.y)));
  const lane = layoutTimeline({ series, width: 1000, start: 10, changes: [{ iter: 20, key: 'driftOn' }] });
  assert.equal(lane.height, 58); assert.ok(lane.yTicks.length > 1); assert.equal(lane.changes.length, 1);
});

test('renderer handles DPR and reveals only the newest segment on explicit calls', () => {
  const c = canvas(), layout = layoutChart({ series, start: 10, previous: series });
  renderChart(c, layout, { dpr: 2, segmentProgress: .5 });
  assert.equal(c.width, 528); assert.equal(c.height, 372);
  assert.equal(c.style.width, '264px'); assert.deepEqual(c.calls[0], ['setTransform', 2, 0, 0, 2, 0, 0]);
  const dot = c.calls.filter(c => c[0] === 'arc').at(-1);
  near(dot[1], (layout.points.at(-2).x + layout.points.at(-1).x) / 2);
  near(dot[2], (layout.points.at(-2).y + layout.points.at(-1).y) / 2);
  assert.equal(c.calls.filter(c => c[0] === 'clearRect').length, 1);
  renderChart(c, layout, { dpr: 1 }); assert.equal(c.width, 264);
  assert.equal(QC_COLORS.accent, '#d55181'); assert.equal(QC_COLORS.mixed, '#f08cbc');
});

test('a scrub glides the selected dot without distorting the run line', () => {
  const c = canvas(), layout = layoutChart({ series, selectedIter: 10 }), from = { x: 50, y: 80 };
  renderChart(c, layout, { segmentProgress: .5, endFrom: from });
  const dot = c.calls.filter(c => c[0] === 'arc').at(-1);
  near(dot[1], (from.x + layout.selected.x) / 2); near(dot[2], (from.y + layout.selected.y) / 2);
  assert.ok(c.calls.some(c => c[0] === 'lineTo' && c[1] === layout.points.at(-1).x && c[2] === layout.points.at(-1).y));
});

test('iterations are whole numbers on the axis; the value beside the dot never covers the axis labels', () => {
  const short = layoutChart({ series: [{ iter: 1, per100: 7 }], width: 220 });
  assert.ok(short.xTicks.length >= 2 && short.xTicks.every(t => Number.isInteger(t.value)), 'no half iterations');
  assert.equal(new Set(short.xTicks.map(t => t.label)).size, short.xTicks.length, 'no label twice ("0 1 1")');
  // A scrubbed early frame: its dot sits at the axis, its value goes to its right.
  const early = layoutChart({ series: [{ iter: 0, per100: 7.2 }, { iter: 100, per100: 9 }], selectedIter: 0 }), c = canvas();
  renderChart(c, early, { dpr: 1 });
  const label = c.calls.find(call => call[0] === 'fillText' && call[1] === '7');
  assert.ok(label[2] > early.selected.x, 'right of a dot at the axis');
  const c2 = canvas(), layout = layoutChart({ series }); renderChart(c2, layout, { dpr: 1 });
  const end = c2.calls.find(call => call[0] === 'fillText' && call[1] === '15');
  assert.ok(end[2] < layout.points.at(-1).x, 'left of the newest dot when there is room');
  // The headline's integer (the engine's `shown`), never the value rounded a second way: 8.6 held at 8 reads 8.
  const held = layoutChart({ series: [{ iter: 10, per100: 5 }, { iter: 20, per100: 8.6, shown: 8 }] }), c3 = canvas();
  renderChart(c3, held, { dpr: 1 });
  assert.ok(c3.calls.some(call => call[0] === 'fillText' && call[1] === '8'));
  assert.ok(!c3.calls.some(call => call[0] === 'fillText' && call[1] === '9'));
});


test('once cells have formed the y range fits the settled line and the start; the forming stretch is drawn faint', () => {
  const run = [...Array.from({ length: 10 }, (_, i) => ({ iter: i * 10, per100: 6 + i * .7, phase: 'burnin' })),
    ...Array.from({ length: 20 }, (_, i) => ({ iter: 100 + i * 10, per100: 13.4 + (i % 3) * .3, phase: 'sampling' }))];
  const layout = layoutChart({ series: run, start: { per100: 14.9 }, formingUntil: 90 });
  assert.deepEqual(layout.yDomain, [10, 18], 'settled 13.4-14.0 and the start 14.9, ±3: a 1-point change is readable');
  assert.ok(layout.points.filter(p => p.forming).length === 10 && layout.points.filter(p => !p.forming).length === 20);
  const early = layoutChart({ series: run.slice(0, 8), start: { per100: 14.9 }, formingUntil: 70 });
  assert.equal(early.yDomain[0], 0, 'while cells form it starts at 0');
  const c = canvas(); renderChart(c, layout, { dpr: 1 });
  const strokes = c.calls.filter(call => call[0] === 'setLineDash').map(call => JSON.stringify(call[1]));
  assert.ok(strokes.includes('[3,3]'), 'the forming stretch dashed (and at 35 %)');
  const label = c.calls.find(call => call[0] === 'fillText' && call[1] === 'Iteration 0: 15');
  assert.ok(label && label[2] < layout.plot.left + 10, 'the start\'s label at the left end, clear of the line\'s end value');
});

test('historical series cannot change the chart, its scale or its hover',()=>{
  const series=[{iter:0,per100:10},{iter:20,per100:12}];
  const own=layoutChart({series}),old=layoutChart({series,previous:[{iter:999,per100:90}]});
  assert.deepEqual(old.xDomain,own.xDomain);assert.deepEqual(old.yDomain,own.yDomain);
  assert.equal(old.previousPoints,undefined);assert.equal(hoverAt(old,old.x(20)).previous,undefined);
});
