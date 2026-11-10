import test from 'node:test';
import assert from 'node:assert/strict';
import { Stage, flashPlan, flashValue, flashNext, FADE_MS, FADE_FRAMES, STATIC_BELOW_MS, FOCUS_IN_MS, FOCUS_OUT_MS, FOCUS_FRAMES } from '../src/stage.js';
import { fakeLoop } from './fake-loop.mjs';

test('flash plan: one static highlight frame for streams and whole-sample runs, a fade of at most 4 frames otherwise', () => {
  const still = { duration: Infinity, fade: false }, fade = { duration: FADE_MS, fade: true };
  for (const interval of [0, 71, 500, 700, STATIC_BELOW_MS - 1]) assert.deepEqual(flashPlan(interval), still, `${interval} ms`);
  for (const interval of [STATIC_BELOW_MS, 5000, Infinity]) assert.deepEqual(flashPlan(interval), fade, `${interval} ms`);
  for (const interval of [0, 700, 5000, Infinity]) assert.deepEqual(flashPlan(interval, true), still, 'a whole-sample run never fades');
  assert.equal(FADE_FRAMES, 4);
  // The static highlight holds 1 until the next snapshot replaces it and never asks for a redraw.
  for (const t of [0, 400, 1e7]) assert.equal(flashValue(still, t), 1);
  assert.equal(flashNext(still, 0), null);
  // The fade steps 1, 2/3, 1/3, 0: four frames, and any redraw in between shows the same value as the step before it.
  const step = FADE_MS / 3;
  assert.deepEqual([0, step, 2 * step, FADE_MS].map(t => flashValue(fade, t)), [1, 1 - 1 / 3, 1 - 2 / 3, 0]);
  assert.equal(flashValue(fade, step - 1), 1); assert.equal(flashValue(fade, step + 10), 1 - 1 / 3);
  assert.deepEqual([0, step, 2 * step].map(t => flashNext(fade, t)), [step, 2 * step, FADE_MS]);
  assert.equal(flashNext(fade, step + 10), 2 * step); assert.equal(flashNext(fade, FADE_MS), null);
  assert.equal(flashValue(null, 0), 0); assert.equal(flashNext(null, 0), null);
});

// The real Stage scheduler (wake/tick/plan/setFrame) on a fake GL context, with draw() recording what it would draw. The
// GL context answers no queries: a getError (a synchronous GPU round trip) would throw.
function harness({ trails = true, layers = [], tiled = false } = {}) {
  const loop = fakeLoop(), draws = [], calls = [];
  const gl = new Proxy({}, { get: (_, k) => k === 'isContextLost' ? () => false : k === 'getError' || k === 'getParameter' || k === 'readPixels' || k === 'checkFramebufferStatus' ? () => { throw new Error(`synchronous ${k}`); }
    : typeof k === 'string' && /^[A-Z0-9_]+$/.test(k) ? 1 : (...args) => { calls.push(k); return {}; } });
  const stage = Object.create(Stage.prototype);
  Object.assign(stage, {
    canvas: { width: 1000, height: 800 }, camera: { x: 0, y: 0, zoom: 1 }, listeners: new Set(), tween: null, raf: null, _dirty: false,
    orbit: { tick() {}, transition: null, active: layers.length > 0, enabled: layers.length > 0, switch() {} },
    options: { trails, cells: true, hover: 0, selected: 0, fill: 0.42 }, gl, limits: { texture: 4096, layers: 16 }, tiled,
    textures: { lab: {}, age: {}, consensus: {}, channels: [] }, layerTexture: {}, layerShape: '', flash: null, flashDue: null,
  });
  stage.draw = function (pulse) { this.dirty = false; draws.push({ t: loop.now, pulse }); };
  const frame = (iter) => ({ nx: 2, ny: 2, x0: 0, y0: 0, voxel: 1, iter, labels: new Uint32Array(4), consensus: null, age: new Uint32Array(4), layers });
  const setFrame = (t, iter, pulse = true) => loop.at(t, () => stage.setFrame(frame(iter), pulse));
  return { loop, draws, stage, setFrame, frame, calls };
}
const between = (draws, a, b) => draws.filter(d => d.t >= a && d.t < b);

test('a live stream (an iteration every 500 ms) draws exactly one frame per iteration, with the static highlight', () => {
  for (const tiled of [false, true]) {
    const h = harness({ tiled });
    for (let k = 0; k < 10; k++) h.setFrame(1000 + 500 * k, k);
    h.loop.run(7000);
    for (let k = 1; k < 10; k++) {
      const t = 1000 + 500 * k, own = between(h.draws, t, t + 500);
      assert.equal(own.length, 1, `iteration ${k}: ${JSON.stringify(own)}`);
      assert.equal(own[0].pulse, 1); assert.ok(own[0].t - t < 17);
    }
    assert.equal(between(h.draws, 5520, 7000).length, 0, 'the last highlight stays until a snapshot replaces it');
    assert.equal(h.loop.frames, h.draws.length, 'every animation frame draws: no idle frame loop');
    assert.ok(h.loop.idle(), 'nothing scheduled between snapshots');
  }
});

test('a whole-sample run shows even a slow iteration as one static frame; a small run fades it in at most 4 frames', () => {
  const tiled = harness({ tiled: true });
  for (const t of [1000, 4000, 9000]) tiled.setFrame(t, t);
  tiled.loop.run(12000);
  assert.deepEqual(tiled.draws.map(d => [d.t - (d.t % 1000), d.pulse]), [[1000, 1], [4000, 1], [9000, 1]]);
  assert.ok(tiled.loop.idle());

  const small = harness();
  small.setFrame(1000, 1);
  small.loop.run(3000);
  const pulses = small.draws.map(d => d.pulse);
  assert.ok(small.draws.length <= FADE_FRAMES, `${small.draws.length} draws`);
  assert.equal(pulses[0], 1, 'drawn at full strength on arrival (the next animation frame)'); assert.equal(pulses.at(-1), 0);
  for (let i = 1; i < pulses.length; i++) assert.ok(pulses[i] < pulses[i - 1], 'the flash decays');
  assert.ok(small.draws.at(-1).t - 1000 < FADE_MS + 20);
  assert.ok(small.loop.idle());
});

test('a jittery stream (0.45-1.4 s apart) never falls back to a fade', () => {
  const h = harness(), times = [];
  for (let k = 0, t = 1000; k < 12; k++, t += k % 2 ? 450 : 1400) { times.push(t); h.setFrame(t, k); }
  h.loop.run(times.at(-1) + 3000);
  for (let k = 1; k < times.length; k++) {
    const own = between(h.draws, times[k], times[k + 1] ?? Infinity);
    assert.equal(own.length, 1, `iteration ${k}: ${own.length} draws`);
  }
  assert.ok(h.loop.idle());
});

test('fast streams (replay, scrubbing) draw once per iteration and nothing after they stop', () => {
  const h = harness();
  for (let k = 0; k < 30; k++) h.setFrame(1000 + 71 * k, k);   // replay at 14 fps
  h.loop.run(4000);
  const streaming = between(h.draws, 1071, 1000 + 71 * 30);
  assert.equal(streaming.length, 29, 'one draw per iteration');
  assert.ok(streaming.every(d => d.pulse === 1));
  assert.equal(between(h.draws, 1000 + 71 * 30, 4000).length, 0, 'the highlight is held, not settled in another frame');
});

test('a redraw for another reason keeps the highlight; a new run starts without one; uploads ask the GPU nothing', () => {
  const h = harness();
  for (let k = 0; k < 3; k++) h.setFrame(1000 + 300 * k, k);
  h.loop.at(2000, () => h.stage.set({ hover: 7 }));   // the pointer rests on a cell between snapshots
  h.loop.run(3000);
  assert.deepEqual(h.draws.filter(d => d.t >= 2000).map(d => d.pulse), [1], 'a hover does not change the highlight');
  h.loop.with(() => h.stage.setFrame(null, false));
  assert.equal(h.stage.flash, null);
  h.loop.with(() => h.stage.setFrame(h.frame(9), false));
  h.loop.run(4000);
  assert.equal(h.draws.at(-1).pulse, 0);
  assert.ok(h.calls.includes('texImage2D'), 'the snapshot was uploaded');   // and no synchronous query was made (it throws)
});

test('camera motion keeps the display rate; trails off, unchanged options and a hidden page draw nothing extra', () => {
  const h = harness();
  h.loop.at(1000, () => h.stage.fit({ x0: 0, y0: 0, w: 100, h: 100 }));
  h.loop.run(2000);
  const tween = between(h.draws, 1000, 1420 + 17);
  assert.ok(tween.length >= 24, `tween drew ${tween.length} frames`);
  assert.ok(h.loop.idle());

  const off = harness({ trails: false });
  off.setFrame(1000, 1);
  off.loop.at(1500, () => off.stage.set({ trails: false, hover: 0 }));
  off.loop.run(3000);
  assert.equal(off.draws.length, 1, 'without trails the flash is invisible: one draw per iteration');

  const hidden = harness();
  hidden.loop.setHidden(true);
  hidden.setFrame(1000, 1);
  hidden.loop.run(3000);
  assert.equal(hidden.draws.length, 0); assert.ok(hidden.loop.idle());
  hidden.loop.with(() => { hidden.loop.hidden = false; hidden.stage.dirty = true; });   // what the visibility handler does
  hidden.loop.run(4000);
  assert.equal(hidden.draws.length, 1); assert.equal(hidden.draws[0].pulse, 0);

  // Covered by Compare's full-screen view (main.js): the same until it goes, then the current state drawn once.
  const covered = harness();
  covered.loop.with(() => covered.stage.setCovered(true));
  covered.setFrame(1000, 1);
  covered.loop.at(1500, () => covered.stage.set({ hover: 3 }));
  covered.loop.run(3000);
  assert.equal(covered.draws.length, 0); assert.ok(covered.loop.idle());
  covered.loop.with(() => covered.stage.setCovered(false));
  covered.loop.run(4000);
  assert.equal(covered.draws.length, 1); assert.ok(covered.loop.idle());
});

test('Go deeper\'s focus eases on the stage\'s own steps (4 frames in over 320 ms, 4 out over 220 ms), then nothing is scheduled; calm: one frame', () => {
  // "Dim eases 0 -> 1 over 320 ms on open and on a new cell, 1 -> 0 over 220 ms on close, through
  // the stage's finite plan (like the tint: a few steps, then idle). Reduced or calm motion: no ease."
  const record = h => { h.stage.draw = function () { this.dirty = false; h.draws.push({ t: h.loop.now, level: this.focusLevel, cell: this.focusShown()?.cell ?? 0 }); }; };
  const h = harness({ trails: false }); record(h);
  h.loop.at(1000, () => h.stage.setFocus({ cell: 7, sources: [[3, 0.4]], same: [] }));
  h.loop.run(2000);
  const into = between(h.draws, 1000, 2000), step = ms => ms / (FOCUS_FRAMES - 1);
  assert.deepEqual(into.map(d => +d.level.toFixed(3)), [0.156, 0.5, 0.844, 1]);
  into.forEach((d, k) => assert.ok(d.t - 1000 >= k * step(FOCUS_IN_MS) && d.t - 1000 < k * step(FOCUS_IN_MS) + 17, `step ${k} at ${d.t}`));
  assert.ok(into.every(d => d.cell === 7)); assert.ok(h.loop.idle(), 'idle once in');
  h.loop.at(3000, () => h.stage.setFocus(null));
  h.loop.run(4000);
  const out = between(h.draws, 3000, 4000);
  assert.deepEqual(out.map(d => +d.level.toFixed(3)), [0.844, 0.5, 0.156, 0]);
  assert.deepEqual(out.map(d => d.cell), [7, 7, 7, 0], 'drawn while it eases out; the last frame draws no focus at all');
  assert.ok(out.at(-1).t - 3000 < FOCUS_OUT_MS + 17); assert.equal(h.stage.focus, null); assert.equal(h.stage.focusOut, null);
  assert.ok(h.loop.idle(), 'idle once out');
  // Calm motion (and reduced motion): one frame each way.
  const calm = harness({ trails: false }); record(calm); calm.stage.calmMotion = true;
  calm.loop.at(1000, () => calm.stage.setFocus({ cell: 7, sources: [] }));
  calm.loop.at(2000, () => calm.stage.setFocus(null));
  calm.loop.run(3000);
  assert.deepEqual(calm.draws.map(d => [d.level, d.cell]), [[1, 7], [0, 0]]); assert.ok(calm.loop.idle());
});

test('3-D: motion renders at reduced resolution, then once more at full resolution when the camera rests', () => {
  const h = harness({ layers: [new Uint32Array(4)] });
  h.setFrame(500, 1, false);
  for (let k = 0; k < 5; k++) h.loop.at(1000 + 16 * k, () => h.stage.panBy(3, 0));
  h.loop.run(3000);
  const last = h.draws.at(-1);
  assert.ok(last.t >= 1064 + 140, 'a refining draw after the motion stops');
  assert.equal(h.stage.wasMoving, false);
  assert.ok(h.loop.idle());
});

test('the 2-D consensus shares the label texture instead of a second upload', () => {
  const h = harness(), uploads = [];
  h.stage.integerTexture = (texture) => uploads.push(texture);
  h.loop.with(() => h.stage.setFrame(h.frame(1), false));
  assert.equal(h.stage.consensusIsLabels, true);
  assert.ok(!uploads.includes(h.stage.textures.consensus));
  const layered = { ...h.frame(2), consensus: new Uint32Array(4) };
  h.loop.with(() => h.stage.setFrame(layered, false));
  assert.equal(h.stage.consensusIsLabels, false);
  assert.ok(uploads.includes(h.stage.textures.consensus));
});
