import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  SuspectGlow, glowBreath, ripplePhases, glowSizes, glowBox, depthScale, GLOW_FPS, GLOW_PERIOD_MS, GLOW_LOW, GLOW_RIPPLE, GLOW_HALO, GLOW_HALO_MIN_PX,
  GLOW_HALO_MAX_PX, GLOW_CORE, GLOW_CORE_LOW, GLOW_COLOR, HALO_STOPS, CORE_STOPS, COUNTERS,
} from '../src/glow.mjs';
import { fakeLoop } from './fake-loop.mjs';

// Go deeper's suspect transcripts, glowing (the one thing that animates by itself):
// its own canvas between the stage and the overlay, sized to the glowing points' box on screen plus the halo and placed
// again on every camera change; in 3-D at each point's depth, as the transcripts are drawn; at most 12 frames a second
// while it is visible, nothing scheduled when it is off, the page hidden or after a lost context; static under reduced
// or calm motion; counted as glowFrames only (never stage draws, never the mixing check's motion).

const SRC = new URL('../src/', import.meta.url);
/** stage.js with its harness hook bound to `hook` (as qc-stage.test.mjs loads it): the stage's own counters. */
async function hookedStage(hook) {
  globalThis.__glowStageHook = hook;
  const dataURL = s => `data:text/javascript;base64,${Buffer.from(s).toString('base64')}`;
  const source = (await readFile(new URL('stage.js', SRC), 'utf8')).replace(/from '\.\/([^']+)'/g, (_, name) =>
    `from ${JSON.stringify(name === 'harness-hook.mjs' ? dataURL('export const harness = globalThis.__glowStageHook;') : new URL(name, SRC).href)}`);
  return import(dataURL(`${source}\n// glow ${Math.random()}`));
}
/** A WebGL2 context that answers what init asks and records nothing (gpu-sync.test.mjs's, quieter). */
function fakeGL() {
  const constants = new Map(), constant = key => { if (!constants.has(key)) constants.set(key, 1000 + constants.size); return constants.get(key); };
  return new Proxy({}, { get(_, key) {
    if (typeof key !== 'string') return undefined;
    if (/^[A-Z0-9_]+$/.test(key)) return constant(key);
    return (...args) => {
      if (key === 'getParameter') return args[0] === constant('ALIASED_POINT_SIZE_RANGE') ? new Float32Array([1, 63]) : 4096;
      if (key === 'getShaderParameter' || key === 'getProgramParameter') return args[1] === constant('ACTIVE_UNIFORMS') ? 0 : true;
      if (key === 'getExtension' || key === 'isContextLost') return key === 'getExtension' ? null : false;
      return {};
    };
  } });
}
/** A 2-D canvas whose context records its calls with the alpha and compositing in force. */
function fakeCanvas() {
  const calls = [], state = { globalAlpha: 1, globalCompositeOperation: 'source-over', fillStyle: '' };
  const g = new Proxy(state, {
    get: (s, k) => k in s ? s[k] : k === 'createRadialGradient' ? (...args) => ({ args, stops: [], addColorStop(at, c) { this.stops.push([at, c]); } })
      : (...args) => { calls.push({ op: k, args, alpha: s.globalAlpha, mode: s.globalCompositeOperation }); },
    set: (s, k, v) => { s[k] = v; return true; } });
  return { width: 0, height: 0, id: '', style: {}, calls, attributes: {}, setAttribute(k, v) { this.attributes[k] = v; }, getContext: () => g };
}
/** The real Stage (the stage's scheduler, draw and counters) on a fake loop and GL, its camera 4 CSS px per µm with
 *  (50, 40) µm in the middle of a 1000 x 800 stage; its glow on fake canvases, counted by `hook` with the times of its
 *  frames. */
async function glowStage({ reduced = false } = {}) {
  const loop = fakeLoop(), counts = {}, times = [], listeners = {};
  const hook = { count(name, n = 1) { counts[name] = (counts[name] || 0) + n; if (name === 'glowFrames' && n) times.push(loop.now); } };
  const { Stage } = await hookedStage(hook);
  const canvas = { width: 0, height: 0, addEventListener(type, fn) { listeners[type] = fn; }, getContext: () => fakeGL(), getBoundingClientRect: () => ({ width: 1000, height: 800 }) };
  const stage = loop.with(() => new Stage(canvas));
  stage.motionQuery = { matches: reduced };
  stage.camera = { x: 50, y: 40, zoom: 4 };
  const canvases = [], createCanvas = () => { const c = fakeCanvas(); canvases.push(c); return c; };
  stage.glowLayer = new SuspectGlow(stage, { dot: () => stage.spillDotPx(), createCanvas, harness: hook });
  loop.run(500);   // the stage's own first frame
  const draws = () => counts.stageDraws ?? 0;
  return { loop, stage, counts, times, listeners, draws, glow: () => canvases.find(c => c.id === 'glow') };
}
const TWO = { key: 'r1:12:a', points: Float32Array.of(50, 40, 0.9, 55, 42, 0.4), giver: [40, 40] };
const since = (times, t) => times.filter(x => x >= t).length;

test('the breath, the ripple and the sizes', () => {
  // Made to glow stronger, then measured in a browser (4 x, 16-36 px still read as small dots):
  // halos about 6 x the dot (30-56 CSS px), a white-pink core of at least 4 px, the breath between 60 % and 100 %.
  assert.deepEqual([GLOW_FPS, GLOW_PERIOD_MS, GLOW_LOW, GLOW_RIPPLE, GLOW_HALO, GLOW_HALO_MIN_PX, GLOW_HALO_MAX_PX, GLOW_CORE, GLOW_CORE_LOW, GLOW_COLOR],
    [12, 1600, 0.6, 0.5, 6, 30, 56, 0.8, 0.85, '#ff4fa3']);
  assert.deepEqual(COUNTERS, ['glowFrames']);
  // The falloffs: the halo hot pink as before (overlaps add up), the core white at its centre through a pale pink
  // (brighter than the old 255,244,250 and 255,190,222), both clear at the rim; frozen (the test corner may draw with them).
  assert.deepEqual(HALO_STOPS.map(s => s[0]), [0, 0.18, 0.42, 0.7, 1]); assert.equal(HALO_STOPS[0][1], 'rgba(255,79,163,1)');
  assert.deepEqual(CORE_STOPS.map(s => s[1]).slice(0, 2), ['rgba(255,255,255,1)', 'rgba(255,218,236,1)']);
  for (const stops of [HALO_STOPS, CORE_STOPS]) { assert.ok(Object.isFrozen(stops) && stops.every(Object.isFrozen)); assert.match(stops.at(-1)[1], /,0\)$/); }
  // Between 60 % and 100 % over 1.6 s, starting low; half a breath behind, the far side is low while the near side peaks.
  assert.equal(glowBreath(0), GLOW_LOW); assert.equal(glowBreath(800), 1); assert.ok(Math.abs(glowBreath(1600) - GLOW_LOW) < 1e-12);
  assert.ok(Math.abs(glowBreath(800, 0.5) - GLOW_LOW) < 1e-12);
  for (let t = 0; t < 3200; t += 37) { const b = glowBreath(t, 0.3); assert.ok(b >= GLOW_LOW - 1e-12 && b <= 1 + 1e-12); }
  // The ripple comes in from the giver's side: phase 0 nearest the giver, GLOW_RIPPLE on the far side.
  const phases = ripplePhases(Float32Array.of(50, 40, 1, 55, 42, 1, 52, 40, 1), [40, 40]);
  assert.equal(phases[0], 0); assert.ok(Math.abs(phases[1] - GLOW_RIPPLE) < 1e-6); assert.ok(phases[2] > 0 && phases[2] < phases[1]);
  assert.deepEqual([...ripplePhases(Float32Array.of(50, 40, 1, 55, 42, 1), null)], [0, 0], 'no giver: in step');
  assert.deepEqual([...ripplePhases(Float32Array.of(50, 40, 1), [50, 40])], [0], 'the giver on the points: in step');
  // The halo: about 6 x the dot, 30-56 CSS px across; the core a little smaller than the dot, at least 4 px.
  assert.deepEqual(glowSizes(4), { halo: 30, core: 4 }); assert.deepEqual(glowSizes(2).halo, 30); assert.equal(glowSizes(7).halo, 42);
  assert.equal(glowSizes(9.5).halo, 56); assert.equal(glowSizes(14).halo, 56); assert.equal(glowSizes(1).core, 4); assert.equal(glowSizes(10).core, 8);
  assert.deepEqual(glowSizes(NaN), glowSizes(6));
  // The box: the points plus their reach, whole CSS px, inside the stage; null off screen.
  assert.deepEqual(glowBox([500, 400, 520, 408], 7, 1000, 800), { x: 493, y: 393, w: 34, h: 22 });
  assert.deepEqual(glowBox([3, 798.5], 7, 1000, 800), { x: 0, y: 791, w: 10, h: 9 }, 'clipped to the stage');
  assert.equal(glowBox([-20, 400], 7, 1000, 800), null); assert.equal(glowBox([], 7, 1000, 800), null);
});

test('it breathes at most 12 times a second in the stage\'s frames, never as stage draws or the mixing check\'s motion, and stops when off', async () => {
  const h = await glowStage();
  assert.equal(h.counts.glowFrames, 0, 'registered at 0 from the start');
  const drawn = h.draws();
  h.loop.at(1000, () => assert.equal(h.stage.setGlow(TWO), true));
  h.loop.run(6000);
  const n = since(h.times, 1000);
  assert.ok(n >= 58 && n <= 61, `${n} frames in 5 s`);
  for (const t of h.times) assert.ok(h.times.filter(u => u >= t && u < t + 1000 - 1e-6).length <= GLOW_FPS, `at ${t}`);
  assert.ok(h.times[0] - 1000 < 17, 'the first one in the next frame');
  assert.equal(h.draws(), drawn, 'the stage itself stayed still'); assert.equal(h.counts.qcAnimFrames ?? 0, 0);
  assert.deepEqual(h.stage.glow, { on: true, count: 2, fps: GLOW_FPS });
  // Breathing: the halos' strength is p x the breath at the frame's time, the far point half a breath behind; the cores'
  // p x (GLOW_CORE_LOW up to 1 with the breath).
  const g = h.glow(), halos = g.calls.filter(c => c.op === 'drawImage' && c.mode === 'lighter').slice(-2), t = h.times.at(-1) - 1000;
  assert.ok(Math.abs(halos[0].alpha - 0.9 * glowBreath(t, 0)) < 1e-6 && Math.abs(halos[1].alpha - 0.4 * glowBreath(t, GLOW_RIPPLE)) < 1e-6);
  const cores = g.calls.filter(c => c.op === 'drawImage' && c.mode === 'source-over').slice(-2);
  assert.ok(Math.abs(cores[0].alpha - 0.9 * (GLOW_CORE_LOW + (1 - GLOW_CORE_LOW) * glowBreath(t, 0))) < 1e-6);
  // Off: hidden at once, no frame after, nothing scheduled, out of the stage's animators.
  h.loop.at(6000, () => assert.equal(h.stage.setGlow(null), true));
  h.loop.run(9000);
  assert.equal(since(h.times, 6000), 0); assert.equal(g.style.display, 'none'); assert.ok(h.loop.idle());
  assert.equal(h.stage.animators.has(h.stage.glowLayer), false);
  assert.deepEqual(h.stage.glow, { on: false, count: 0, fps: 0 });
});

test('the same glow again is no work; points without a place or a likelihood do not glow; p is at most 1', async () => {
  const h = await glowStage({ reduced: true });
  h.loop.at(1000, () => h.stage.setGlow(TWO));
  h.loop.run(2000);
  assert.equal(since(h.times, 1000), 1, 'static: one frame');
  h.loop.with(() => {
    assert.equal(h.stage.setGlow({ ...TWO }), false, 'the same key, points and giver');
    assert.equal(h.stage.setGlow({ ...TWO, points: Float32Array.of(1, 2, 0.5, 3, 4, 0.5) }), false, 'the key names the answer: no work');
  });
  h.loop.run(3000);
  assert.equal(since(h.times, 1000), 1); assert.ok(h.loop.idle());
  h.loop.at(3000, () => assert.equal(h.stage.setGlow({ ...TWO, giver: null }), true, 'another giver'));
  h.loop.run(4000);
  assert.equal(since(h.times, 3000), 1);
  h.loop.at(4000, () => h.stage.setGlow({ key: 'b', points: Float32Array.of(50, 40, 3, NaN, 40, 1, 52, 41, 0, 53, 41, -1, 54, 41, NaN), giver: null }));
  h.loop.run(5000);
  assert.equal(h.stage.glow.count, 1);
  assert.deepEqual(h.glow().calls.filter(c => c.op === 'drawImage').slice(-2).map(c => c.alpha), [1, 1], 'p 3 glows as 1');
  h.loop.at(5000, () => h.stage.setGlow({ key: 'c', points: Float32Array.of(50, 40, 0), giver: null }));
  h.loop.run(6000);
  assert.deepEqual(h.stage.glow, { on: false, count: 0, fps: 0 }, 'nothing to glow: off'); assert.ok(h.loop.idle());
  h.loop.with(() => {
    assert.equal(h.stage.setGlow(null), true); assert.equal(h.stage.setGlow(null), false);
    assert.equal(h.stage.setGlow({ key: 'd', points: [50, 40, 1] }), false, 'points must be a Float32Array: none, as before');
  });
});

test('its canvas: the glowing points\' box plus the halo, between the stage and the overlay, placed again on every camera change', async () => {
  const h = await glowStage();
  // Between the stage's canvas and the overlay: inserted right after the stage's canvas in its parent.
  const parent = { inserted: [], insertBefore(node, before) { this.inserted.push([node, before]); } };
  h.stage.canvas.parentNode = parent; h.stage.canvas.nextSibling = 'overlay';
  h.loop.at(1000, () => h.stage.setGlow(TWO));
  h.loop.run(1050);
  const c = h.glow();
  assert.deepEqual(parent.inserted, [[c, 'overlay']]);
  assert.equal(c.attributes['aria-hidden'], 'true'); assert.match(c.style.cssText, /position:absolute;.*pointer-events:none/);
  assert.match(c.style.cssText, /mix-blend-mode:screen/, 'it only adds light (screen), so it brightens on bright stain too');
  // (50, 40) and (55, 42) µm are at (500, 400) and (520, 408) CSS px; the dot is 4 px (its least), the halo 30 px across
  // (its least since the browser run of 2026-10-22), so the box reaches 16 px around them.
  assert.deepEqual([c.style.left, c.style.top, c.style.width, c.style.height, c.width, c.height, c.style.display], ['484px', '384px', '52px', '40px', 52, 40, 'block']);
  const last = () => c.calls.slice(c.calls.findLastIndex(x => x.op === 'clearRect'));
  assert.deepEqual(last().find(x => x.op === 'setTransform').args, [1, 0, 0, 1, -484, -384]);
  const halos = last().filter(x => x.op === 'drawImage' && x.mode === 'lighter').map(x => x.args.slice(1));
  assert.deepEqual(halos, [[485, 385, 30, 30], [505, 393, 30, 30]], 'a halo centred on each point, in CSS px of the stage');
  const cores = last().filter(x => x.op === 'drawImage' && x.mode === 'source-over').map(x => x.args.slice(1).map(v => +v.toFixed(2)));
  assert.deepEqual(cores, [[498, 398, 4, 4], [518, 406, 4, 4]], 'a bright core over each, 4 px at the least');
  // A pan: placed again and drawn in that frame (with the stage), not on the next breath.
  h.loop.run(1199);
  const frames = h.times.length;
  h.loop.at(1200, () => h.stage.panBy(10, 5));
  h.loop.run(1218);
  assert.deepEqual([c.style.left, c.style.top], ['494px', '389px']); assert.equal(h.times.length, frames + 1);
  // Zoomed in: the dot grows, the halo with it (at most 56 px).
  h.loop.at(1300, () => h.stage.zoomAt(500, 400, 4));
  h.loop.run(1320);
  assert.equal(last().find(x => x.op === 'drawImage').args[3], 56, 'the dot at 9.6 px: a 56 px halo (6 x 9.6 is over its most)');
  // Off screen: hidden, and nothing scheduled until the camera brings it back.
  h.loop.at(2000, () => h.stage.panBy(-5000, 0));
  h.loop.run(4000);
  assert.equal(c.style.display, 'none'); assert.equal(since(h.times, 2020), 0); assert.ok(h.loop.idle());
  assert.equal(h.stage.glow.on, false);
  h.loop.at(4000, () => h.stage.panBy(5000, 0));
  h.loop.run(4100);
  assert.equal(c.style.display, 'block'); assert.ok(since(h.times, 4000) >= 1);
});

test('nothing is scheduled while the page is hidden, in 3-D or with the context lost; it breathes again after', async () => {
  const h = await glowStage();
  h.loop.at(1000, () => h.stage.setGlow(TWO));
  h.loop.run(2000);
  h.loop.setHidden(true);   // the stage's own visibility handler sleeps
  h.loop.run(5000);
  assert.equal(since(h.times, 2000), 0); assert.ok(h.loop.idle(), 'hidden: nothing scheduled');
  h.loop.setHidden(false);
  h.loop.run(6000);
  assert.ok(since(h.times, 5000) >= 10, 'shown again: it breathes');
  // 3-D with points that carry no depth (TWO has none): not drawn there, hidden in the next frame, then nothing.
  h.loop.at(6000, () => { h.stage.orbit.enabled = true; h.stage.dirty = true; });
  h.loop.run(8000);
  assert.equal(h.glow().style.display, 'none'); assert.equal(since(h.times, 6000), 0); assert.ok(h.loop.idle());
  assert.equal(h.stage.glow.on, false);
  h.loop.at(8000, () => { h.stage.orbit.enabled = false; h.stage.dirty = true; });
  h.loop.run(9000);
  assert.equal(h.glow().style.display, 'block'); assert.ok(since(h.times, 8000) >= 10);
  // A lost context: hidden at once, nothing scheduled; restored: drawn again.
  h.loop.at(9000, () => h.listeners.webglcontextlost({ preventDefault() {} }));
  h.loop.run(11000);
  assert.equal(h.glow().style.display, 'none'); assert.equal(since(h.times, 9000), 0); assert.ok(h.loop.idle());
  h.loop.at(11000, () => h.listeners.webglcontextrestored());
  h.loop.run(12000);
  assert.equal(h.glow().style.display, 'block'); assert.ok(since(h.times, 11000) >= 10);
  // Covered by Compare's full-screen view (main.js stage.setCovered): nothing scheduled; uncovered: it breathes again.
  h.loop.at(12000, () => h.stage.setCovered(true));
  h.loop.run(14000);
  assert.equal(since(h.times, 12000), 0); assert.ok(h.loop.idle(), 'covered: nothing scheduled');
  h.loop.at(14000, () => h.stage.setCovered(false));
  h.loop.run(15000);
  assert.ok(since(h.times, 14000) >= 10, 'uncovered: it breathes');
});

test('static under reduced or calm motion: one frame per change, at full strength, then nothing', async () => {
  const r = await glowStage({ reduced: true });
  r.loop.at(1000, () => r.stage.setGlow(TWO));
  r.loop.run(5000);
  assert.equal(since(r.times, 1000), 1); assert.ok(r.loop.idle());
  assert.deepEqual(r.glow().calls.filter(c => c.op === 'drawImage').map(c => c.alpha), [0.9, 0.4, 0.9, 0.4].map(v => Math.fround(v)), 'p, no breath');
  assert.deepEqual(r.stage.glow, { on: true, count: 2, fps: 0 });
  r.loop.at(5000, () => r.stage.panBy(10, 0));
  r.loop.run(7000);
  assert.equal(since(r.times, 5000), 1, 'a camera change: one more'); assert.ok(r.loop.idle());
  // The calm switch while it breathes: one frame at full strength, then still; off again: it breathes.
  const c = await glowStage();
  c.loop.at(1000, () => c.stage.setGlow(TWO));
  c.loop.run(2000);
  c.loop.at(2000, () => { c.stage.calmMotion = true; c.stage.replan(); });
  c.loop.run(5000);
  assert.ok(since(c.times, 2000) <= 1); assert.ok(c.loop.idle()); assert.equal(c.stage.glow.fps, 0);
  assert.deepEqual(c.glow().calls.filter(x => x.op === 'drawImage').slice(-4).map(x => x.alpha), [0.9, 0.4, 0.9, 0.4].map(v => Math.fround(v)));
  c.loop.at(5000, () => { c.stage.calmMotion = false; c.stage.replan(); });
  c.loop.run(6000);
  assert.ok(since(c.times, 5000) >= 10);
});

// ---- in 3-D: at each point's depth ---------------------------------------------------------------------------------------
const DEG55 = 55 * Math.PI / 180;
/** The glow stage turned to 3-D at 55 degrees (`yaw`), a 15 µm slab at its full height. */
const into3d = (h, yaw = 0) => { h.stage.info = { z: { thickness_um: 15 } }; Object.assign(h.stage.orbit, { enabled: true, angle: DEG55, yaw, amount: 1 }); h.stage.dirty = true; };
const DEEP = { ...TWO, key: 'r1:12:3d', depth: Float32Array.of(0.5, 0.2) };
/** The halos and cores of the glow's last frame: [centre x, centre y, size] each (CSS px of the stage). */
const lastFrame = c => {
  const calls = c.calls.slice(c.calls.findLastIndex(x => x.op === 'clearRect'));
  const of = mode => calls.filter(x => x.op === 'drawImage' && x.mode === mode).map(x => [x.args[1] + x.args[3] / 2, x.args[2] + x.args[4] / 2, x.args[3]]);
  return { halos: of('lighter'), cores: of('source-over') };
};

test('in 3-D it glows at each point\'s depth: placed by Orbit.project (the shaders\' own), sized by distance as the transcripts\' dots, breathing at most 12 times a second', async () => {
  const h = await glowStage();
  h.loop.at(1000, () => { into3d(h); assert.equal(h.stage.setGlow(DEEP), true); });
  h.loop.run(3000);
  const n = since(h.times, 1000);
  assert.ok(n >= 22 && n <= 25, `${n} frames in 2 s`);
  for (const t of h.times.filter(t => t >= 1000)) assert.ok(h.times.filter(u => u >= t && u < t + 1000 - 1e-6).length <= GLOW_FPS, `at ${t}`);
  assert.deepEqual(h.stage.glow, { on: true, count: 2, fps: GLOW_FPS });
  const { halos, cores } = lastFrame(h.glow()), o = h.stage.orbit, dist = o.distance(), { halo, core } = glowSizes(h.stage.spillDotPx());
  [[50, 40, 0.5], [55, 42, 0.2]].forEach(([x, y, z], i) => {
    const p = o.project(x, y, z), k = depthScale(dist, p.depth);
    assert.ok(Math.abs(halos[i][0] - p.x) < 1e-3 && Math.abs(halos[i][1] - p.y) < 1e-3, `halo ${i} at ${halos[i]} against ${p.x},${p.y}`);
    assert.ok(Math.abs(cores[i][0] - p.x) < 1e-3 && Math.abs(cores[i][1] - p.y) < 1e-3, `core ${i}`);
    assert.ok(Math.abs(halos[i][2] - halo * k) < 1e-6 && Math.abs(cores[i][2] - core * k) < 1e-6, `sizes ${i}: x ${k}`);
    assert.ok(k >= 0.85 && k <= 1.5);
  });
  // Raised off the image: not where the 2-D view would put it (the depth counts).
  const flat = h.stage.toScreen(50, 40);
  assert.ok(Math.hypot(halos[0][0] - flat[0], halos[0][1] - flat[1]) > 5, 'its 3-D place');
  // The transcripts' rule for the dots' size by depth.
  assert.deepEqual([depthScale(100, 100), depthScale(100, 400), depthScale(100, 10), depthScale(100, 64)], [1, 0.85, 1.5, 1.25]);
});

test('in 3-D each kept point keeps its own depth (a point dropped for no place or no likelihood takes its depth with it); a thicker slab repaints it where the view now projects it', async () => {
  const r = await glowStage({ reduced: true });
  const points = Float32Array.of(50, 40, 0.9, 52, 41, 0, 55, 42, 0.4, NaN, 40, 0.8, 58, 44, 0.7), depth = Float32Array.of(0.5, 0.9, 0.2, 0.7, 0.8);
  r.loop.at(1000, () => { into3d(r); r.stage.setGlow({ key: 'dropped', points, depth, giver: null }); });
  r.loop.run(2000);
  assert.equal(r.stage.glow.count, 3);
  const check = what => {
    const halos = lastFrame(r.glow()).halos;
    assert.equal(halos.length, 3, what);
    [[50, 40, 0.5], [55, 42, 0.2], [58, 44, 0.8]].forEach(([x, y, z], i) => {
      const p = r.stage.orbit.project(x, y, z);
      assert.ok(Math.abs(halos[i][0] - p.x) < 1e-3 && Math.abs(halos[i][1] - p.y) < 1e-3, `${what}: halo ${i} at ${halos[i]} against ${p.x},${p.y}`);
    });
  };
  check('each at its own depth');
  // The slab twice as thick stands the points twice as high: one repaint, each halo where the view projects it now.
  const before = lastFrame(r.glow()).halos;
  r.loop.at(2000, () => { r.stage.info = { z: { thickness_um: 30 } }; r.stage.dirty = true; });
  r.loop.run(3000);
  assert.equal(since(r.times, 2000), 1, 'one frame for the new slab');
  check('on the thicker slab'); assert.notDeepEqual(lastFrame(r.glow()).halos, before);
});

test('in 3-D a turn of the view repaints it in that frame; still, under reduced or calm motion, one frame per change; a point without a depth only in 2-D', async () => {
  const r = await glowStage({ reduced: true });
  r.loop.at(1000, () => { into3d(r); r.stage.setGlow(DEEP); });
  r.loop.run(3000);
  assert.equal(since(r.times, 1000), 1, 'one frame'); assert.ok(r.loop.idle());
  assert.deepEqual(r.stage.glow, { on: true, count: 2, fps: 0 });
  const before = lastFrame(r.glow()).halos;
  // The view turned (an orbit drag: yaw 0.6): repainted once, each halo where the new view projects its point.
  r.loop.at(3000, () => { r.stage.orbit.yaw = 0.6; r.stage.dirty = true; r.stage.emit(); });
  r.loop.run(5000);
  assert.equal(since(r.times, 3000), 1, 'one more for the turn'); assert.ok(r.loop.idle());
  const after = lastFrame(r.glow()).halos, p = r.stage.orbit.project(50, 40, 0.5);
  assert.ok(Math.abs(after[0][0] - p.x) < 1e-3 && Math.abs(after[0][1] - p.y) < 1e-3); assert.notDeepEqual(after, before);
  // The slab's height (its stretch) moves the points too.
  r.loop.at(5000, () => { r.stage.set({ stretch: 3 }); });
  r.loop.run(6000);
  assert.equal(since(r.times, 5000), 1);
  // One point with a depth, one without: only the first is drawn in 3-D; both in 2-D.
  r.loop.at(6000, () => { r.stage.setGlow({ ...DEEP, key: 'half', depth: Float32Array.of(0.5, NaN) }); });
  r.loop.run(7000);
  assert.equal(lastFrame(r.glow()).halos.length, 1); assert.equal(r.stage.glow.count, 2, 'both still glow (the window\'s count)');
  r.loop.at(7000, () => { Object.assign(r.stage.orbit, { enabled: false, angle: 0, yaw: 0, amount: 0 }); r.stage.dirty = true; });
  r.loop.run(8000);
  const flat = lastFrame(r.glow()).halos;
  assert.equal(flat.length, 2, '2-D: every point');
  assert.deepEqual(flat.map(([x, y]) => [x, y]), [r.stage.toScreen(50, 40), r.stage.toScreen(55, 42)]);
  // None with a depth in 3-D: nothing drawn, nothing scheduled (the 2-D view draws them again).
  r.loop.at(8000, () => { into3d(r); r.stage.setGlow({ ...TWO, key: 'none' }); });
  r.loop.run(10000);
  assert.equal(r.glow().style.display, 'none'); assert.equal(r.stage.glow.on, false); assert.ok(r.loop.idle());
});
