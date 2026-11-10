import test from 'node:test';
import assert from 'node:assert/strict';
import { PaintBatch, HOLD_MS } from '../src/paint-batch.mjs';
import { Timeline, REDRAW_MS } from '../src/timeline.js';
import { fakeLoop, fakeElement } from './fake-loop.mjs';

// The panels batch on a virtual 60 Hz display: which animation frame each job runs in.
function harness() {
  const loop = fakeLoop(), batch = loop.with(() => new PaintBatch()), ran = [];
  const job = name => () => ran.push({ name, t: loop.now, frame: loop.frames });
  return { loop, batch, ran, job };
}

test('marks made at different times within a frame run together, once per key, in the order first marked', () => {
  const h = harness();
  h.loop.at(100, () => { h.batch.mark('hud', h.job('hud-old')); h.batch.mark('facts', h.job('facts')); });
  h.loop.at(105, () => h.batch.mark('hud', h.job('hud')));   // the latest function of a key replaces the earlier one
  h.loop.run(1000);
  assert.deepEqual(h.ran.map(r => r.name), ['hud', 'facts']);
  assert.equal(new Set(h.ran.map(r => r.frame)).size, 1, 'one frame'); assert.equal(h.batch.flushes, 1);
  assert.ok(h.loop.idle(), 'nothing scheduled once flushed');
});

test('a snapshot on its way holds the batch: panels marked meanwhile run in the frame its upload is drawn in', () => {
  const h = harness();
  let token, drawnIn = null;
  h.loop.at(100, () => { h.batch.mark('run', h.job('status: 6 snapshots')); token = h.batch.hold(); });   // WS message, then show()
  h.loop.at(150, () => h.batch.mark('now', h.job('statistics')));                                      // the frame's statistics
  h.loop.at(320, () => {                                                                               // the frame arrives
    h.batch.mark('hud', h.job('hud'));
    requestAnimationFrame(() => { drawnIn = h.loop.frames; });   // the stage draws the uploaded frame
    h.batch.release(token);
  });
  h.loop.run(1000);
  assert.deepEqual(h.ran.map(r => r.name), ['status: 6 snapshots', 'statistics', 'hud']);
  assert.ok(h.ran.every(r => r.t >= 320 && r.frame === drawnIn), JSON.stringify(h.ran));
  assert.equal(h.batch.flushes, 1); assert.ok(h.loop.idle());
});

test('released with its upload, a snapshot writes its panels in that task, even if the next snapshot holds at once', () => {
  const h = harness();
  let drawnIn = null;
  h.loop.at(100, () => { const token = h.batch.hold(); h.batch.mark('run', h.job('status')); h.loop.at(250, () => {
    h.batch.mark('hud', h.job('hud'));
    requestAnimationFrame(() => { drawnIn = h.loop.frames; });       // the stage draws the upload in the next frame
    h.batch.release(token, true);
    h.batch.hold();                                                  // follow() fetches the next snapshot right away
  }); });
  h.loop.run(2000);
  assert.deepEqual(h.ran.map(r => [r.name, r.t]), [['status', 250], ['hud', 250]]);
  assert.ok(h.ran.every(r => r.frame + 1 === drawnIn), 'written before the frame that draws the upload');
  assert.equal(h.batch.flushes, 1);
});

test('a hold expires after HOLD_MS; overlapping holds wait for the last; urgent marks are never held', () => {
  const h = harness();
  h.loop.at(100, () => { h.batch.hold(); h.batch.mark('hud', h.job('hud')); });   // a fetch that never returns
  h.loop.run(2000);
  assert.equal(h.ran.length, 1); assert.ok(h.ran[0].t >= 100 + HOLD_MS && h.ran[0].t < 100 + HOLD_MS + 20, `${h.ran[0].t}`);

  const o = harness();
  let a, b;
  o.loop.at(100, () => { a = o.batch.hold(); b = o.batch.hold(); o.batch.mark('hud', o.job('hud')); });
  o.loop.at(200, () => o.batch.release(a));
  o.loop.at(300, () => o.batch.release(b));
  o.loop.run(1000);
  assert.equal(o.ran.length, 1); assert.ok(o.ran[0].t >= 300 && o.ran[0].t < 320);

  const u = harness();
  u.loop.at(100, () => { u.batch.hold(); u.batch.mark('hud', u.job('hud')); u.batch.mark('timeline', u.job('hover'), true); });
  u.loop.run(130);
  assert.deepEqual(u.ran.map(r => r.name), ['hud', 'hover'], 'the user is interacting: everything queued runs now');
});

test('covered by a full-screen view, nothing is asked for: the jobs wait (the latest per key) and run once uncovered', () => {
  const h = harness();
  h.loop.at(100, () => { h.batch.mark('run', h.job('run-old')); h.batch.setCovered(true); });   // a batch was asked for
  h.loop.at(150, () => { h.batch.mark('run', h.job('run')); h.batch.mark('runs', h.job('runs')); });
  h.loop.run(1000);
  assert.equal(h.ran.length, 0); assert.equal(h.loop.frames, 0, 'no animation frame at all'); assert.ok(h.loop.idle());
  h.loop.at(1100, () => h.batch.setCovered(false));
  h.loop.run(2000);
  assert.deepEqual(h.ran.map(r => r.name), ['run', 'runs']); assert.equal(h.batch.flushes, 1); assert.ok(h.loop.idle());
});

test('covered, a snapshot released with its upload writes nothing: its panels wait for the batch once uncovered', () => {
  // A paused tiled run's new window, or a snapshot already on its way, lands under Compare.
  const h = harness();
  let token;
  h.loop.at(100, () => { token = h.batch.hold(); h.batch.setCovered(true); });   // show() fetching; Compare opens
  h.loop.at(300, () => {                                                         // the snapshot lands under the dialog
    for (const name of ['run', 'now', 'hud', 'qc']) h.batch.mark(name, h.job(name));
    h.batch.ride('timeline', h.job('timeline'));
    h.batch.release(token, true);
  });
  h.loop.run(1000);
  assert.equal(h.ran.length, 0); assert.equal(h.batch.flushes, 0); assert.ok(h.batch.pending);
  assert.equal(h.loop.frames, 0, 'no animation frame at all'); assert.ok(h.loop.idle());
  h.loop.at(1100, () => h.batch.setCovered(false));                              // Back
  h.loop.run(2000);
  assert.deepEqual(h.ran.map(r => r.name), ['run', 'now', 'hud', 'qc', 'timeline']);
  assert.ok(h.ran.every(r => r.t >= 1100)); assert.equal(h.batch.flushes, 1); assert.ok(h.loop.idle());
  // Uncovered, a release with the upload writes at once again.
  h.loop.at(2100, () => { const t = h.batch.hold(); h.batch.mark('hud', h.job('hud-2')); h.batch.release(t, true); });
  h.loop.run(2101);
  assert.deepEqual(h.ran.at(-1), { name: 'hud-2', t: 2100, frame: h.loop.frames });
});

test('a rider joins the next batch without asking for one; a failing job does not stop the others', () => {
  const h = harness();
  h.loop.at(100, () => h.batch.ride('timeline', h.job('timeline')));
  h.loop.run(500);
  assert.equal(h.ran.length, 0); assert.ok(h.loop.idle(), 'a rider alone schedules nothing');
  h.loop.at(600, () => h.batch.mark('hud', h.job('hud')));
  h.loop.run(1000);
  assert.deepEqual(h.ran.map(r => r.name), ['hud', 'timeline']);
  const errors = [], error = console.error;
  console.error = e => errors.push(e.message);
  try {
    h.loop.at(1100, () => { h.batch.mark('a', () => { throw new Error('broken panel'); }); h.batch.mark('b', h.job('b')); });
    h.loop.run(1200);
  } finally { console.error = error; }
  assert.deepEqual(errors, ['broken panel']); assert.equal(h.ran.at(-1).name, 'b');
});

test('with the batch, a live timeline draws in the snapshot frames; a throttled redraw rides with the next one', () => {
  const loop = fakeLoop(), batch = loop.with(() => new PaintBatch()), canvas = fakeElement({ clientWidth: 1000, clientHeight: 116 });
  const timeline = loop.with(() => new Timeline(canvas, { onScrub() {}, batch })), draws = [], frames = [];
  const draw = timeline.draw.bind(timeline);
  timeline.draw = () => { draws.push(loop.frames); draw(); };
  loop.with(() => timeline.setPlan([{ phase: 'burnin', frames: null }]));
  loop.run(500);
  draws.length = 0;
  const list = [];
  for (let k = 0; k < 20; k++) loop.at(1000 + 33 * k, () => {   // 30 snapshots a second: each one a held batch
    list.push({ seq: k, iter: k, phase: 'burnin', changed: k, loglik: -k, live: {} });
    const token = batch.hold(); timeline.setFrames(list); timeline.setCurrent(k);
    loop.at(loop.now + 10, () => { requestAnimationFrame(() => frames.push(loop.frames)); batch.release(token); });
  });
  loop.run(3000);
  assert.ok(draws.length >= 6 && draws.length <= 20, `${draws.length} draws`);
  assert.ok(draws.every(f => frames.includes(f)), 'every timeline draw shares a snapshot frame');
  for (let i = 1; i < draws.length; i++) assert.ok(draws[i] > draws[i - 1]);
  assert.equal(timeline.current, 19); assert.equal(timeline.dirty, false, 'the last state is drawn');
  assert.ok(REDRAW_MS > 0); assert.ok(loop.idle());
});
