import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  Stage, MASK_FS, XENIUM_FS, COLORS, MIXING_CODES, MIXING_COLORS, MIX_WIDTH, MIX_UNITS, TINT_MS, TINT_FRAMES, TINT_AFTER_MS,
  mixingStep, packMixing, tintPlan, tintValue, tintNext, MixingTextures, ColourGPU, pointColourBy,
} from '../src/stage.js';
import { CLUSTER_GLSL } from '../src/cluster-gpu.js';
import { volumeFragment } from '../src/volume.js';
import { pointVertex } from '../src/transcripts.js';
import { fakeLoop } from './fake-loop.mjs';

// Colour by: Mixing in the stage: the shader sources, the per-label texture
// in the cluster lookup's layout, the binding every program built on the shared colour functions gets, and the
// cross-fade that runs only for results a second or more apart, in at most 3 frames, on the stage's own scheduler.

const SRC = new URL('../src/', import.meta.url);
const ANCHORS = ['uniform bool uHasAge;', 'vec3 base = displayColor(id);', 'col.a *= clusterWeight(id);'];
const occurrences = (text, part) => text.split(part).length - 1;

test('the three XENIUM_FS anchors stay byte-identical in the mask shader, once each, and every replacement applies', async () => {
  const source = await readFile(new URL('stage.js', SRC), 'utf8');
  const xenium = source.slice(source.indexOf('export const XENIUM_FS'));
  assert.deepEqual([...xenium.matchAll(/\.replace\('([^']*)'/g)].map(m => m[1]), ANCHORS, 'the replace() anchors themselves');
  for (const anchor of ANCHORS) assert.equal(occurrences(MASK_FS, anchor), 1, anchor);
  assert.ok(XENIUM_FS.includes('uniform bool uHasAge; uniform usampler2D uMapping; uniform int uMapWidth; uniform float uOpacity;'));
  assert.ok(XENIUM_FS.includes('vec3 base = mapped == 0u ? vec3(.55) : displayColor(mapped);'));
  assert.ok(XENIUM_FS.includes('col.a *= uOpacity * (mapped == 0u ? 1. : clusterWeight(mapped));'));
  assert.equal(occurrences(XENIUM_FS, 'vec3 base = displayColor(id);'), 0); assert.equal(occurrences(XENIUM_FS, 'col.a *= clusterWeight(id);'), 0);
  assert.ok(XENIUM_FS.includes('fill *= stripes; edgeAlpha *= stripes;'), 'the Xenium masks get the stripes of their mapped cell too');
});

test('Go deeper\'s focus reaches the Xenium masks only through the veil: their shader keeps its gate shut', async () => {
  // XENIUM_FS inherits the mask's greying line behind uFocusOn; xenium-masks.js never sets a focus uniform, so it stays
  // false there (the veil drawn after them darkens them instead), and the shared colour functions know nothing of it.
  const source = await readFile(new URL('xenium-masks.js', SRC), 'utf8');
  assert.equal(/uFocus/.test(source), false);
  assert.ok(XENIUM_FS.includes('if (uFocusOn && !focusLit(id)) col = vec4('));
  assert.equal(/uFocus|focusLit/.test(COLORS), false);
  for (const anchor of ANCHORS) assert.equal(occurrences(MASK_FS, anchor), 1, anchor);
});

test('the mixing branch lives in the colour functions every view shares, and never reads gl_FragCoord there', () => {
  assert.ok(CLUSTER_GLSL.includes('cellColor(label)'), 'displayColor() falls back to cellColor(): the mixing branch is reached through it');
  assert.ok(COLORS.includes(CLUSTER_GLSL), 'the cluster functions are unchanged');
  assert.ok(COLORS.includes('vec3 cellColor(uint id) { return uMixing ? mixingColor(id) : identityColor(id); }'));
  for (const name of ['uniform highp sampler2D uMixFrom, uMixTo;', 'uniform bool uMixing;', 'uniform float uMixT;', 'uniform ivec2 uMixCount;', 'float mixStripes = 0.;'])
    assert.ok(COLORS.includes(name), name);
  assert.ok(COLORS.indexOf('vec3 identityColor') < COLORS.indexOf('vec3 cellColor') && COLORS.indexOf('vec3 cellColor') < COLORS.indexOf('vec3 displayColor'));
  assert.equal(COLORS.includes('gl_FragCoord'), false, 'COLORS also runs in the transcripts\' vertex shader');
  for (const shader of [volumeFragment(COLORS), pointVertex(COLORS), MASK_FS, XENIUM_FS]) assert.ok(shader.includes('mixingColor(uint label)'));
  // The stripes: fragment-only, outside COLORS, applied to the fill and to the outline (Outlines mode keeps the cue),
  // after the cell's displayColor set mixStripes and before either is used.
  assert.equal(COLORS.includes('float mixingFill()'), false); assert.ok(MASK_FS.indexOf('float mixingFill()') >= MASK_FS.indexOf(COLORS) + COLORS.length);
  const stripes = MASK_FS.indexOf('float stripes = mixingFill();'), applied = MASK_FS.indexOf('fill *= stripes; edgeAlpha *= stripes;');
  assert.ok(stripes > MASK_FS.indexOf('vec3 base = displayColor(id);') && stripes > MASK_FS.indexOf('if (dimmed)') && applied > stripes);
  assert.ok(applied < MASK_FS.indexOf('col = vec4(base, fill);') && applied < MASK_FS.indexOf('col = vec4(edgeColor, edgeAlpha);'));
  assert.equal(occurrences(MASK_FS, 'mixingFill()'), 2, 'defined once, called once');
  assert.ok(/mod\(gl_FragCoord\.x \+ gl_FragCoord\.y, 8\.\) < 4\. \? \.7 : \.25/.test(MASK_FS), 'diagonal bands fixed to the screen');
  // Every other cell and colour mode: exactly 1, so the fill and the outline are what they were (parity with QC off).
  assert.ok(MASK_FS.includes('if (mixStripes <= 0.) return 1.;'));
});

test('the display modes are Cells (0) and Outlines (1): no Activity heat left in the 2-D mask, the Xenium masks or the 3-D volume', () => {
  for (const shader of [MASK_FS, XENIUM_FS, volumeFragment(COLORS)]) {
    assert.doesNotMatch(shader, /heat|activity|uMode\s*==\s*2/i);
    assert.match(shader, /uniform int uMode;/);
  }
  // Cells and Outlines draw as they did: the fill (Cells) or none (Outlines), the same outline strength.
  assert.ok(MASK_FS.includes('float fill = uMode == 0 ? uFill : 0.0;') && MASK_FS.includes('float edgeAlpha = 0.95;'));
  assert.match(volumeFragment(COLORS), /vec3 base = displayColor\(id\);/);
  // The shared colour functions end with the cluster functions (and the line that ends them), as before.
  assert.ok(COLORS.startsWith('vec3 hsv') && COLORS.endsWith(`${CLUSTER_GLSL}\n`));
  assert.ok(MASK_FS.slice(MASK_FS.indexOf(COLORS) + COLORS.length).startsWith('// Not judged, under Colour by: Mixing'));
});

test('the shader\'s ramp is the one-hue palette and steps at 10 % and 30 % (codes 25 and 75), 255 = not judged', () => {
  assert.deepEqual({ ...MIXING_COLORS }, { little: '#6b3f63', some: '#b2477a', mixed: '#f08cbc', unjudged: '#5c6573', accent: '#d55181' });
  assert.deepEqual({ ...MIXING_CODES }, { scale: 250, some: 25, mixed: 75, unjudged: 255 });
  const body = COLORS.slice(COLORS.indexOf('vec3 mixingStep(int code)'), COLORS.indexOf('vec3 mixingColor'));
  const vec = hex => `vec3(${[1, 3, 5].map(i => (parseInt(hex.slice(i, i + 2), 16) / 255).toFixed(4)).join(', ')})`;
  assert.equal(body, `vec3 mixingStep(int code) {
  return code >= 255 ? ${vec('#5c6573')} : code >= 75 ? ${vec('#f08cbc')}
    : code >= 25 ? ${vec('#b2477a')} : ${vec('#6b3f63')};
}
`);
  assert.ok(COLORS.includes(`ivec2(int(label) % ${MIX_WIDTH}, int(label) / ${MIX_WIDTH})`), 'the cluster lookup\'s 4096-wide layout');
  assert.ok(COLORS.includes('if (int(label) >= count) return 256;'), 'a label past the result: no result (grey, no stripes)');
  // 0.30 codes as floor(0.30 x 250) = 75 and 0.10 as 25: the steps are exactly the deck's.
  assert.deepEqual([0, 24, 25, 74, 75, 250, 254, 255, null, undefined].map(mixingStep),
    ['little', 'little', 'some', 'some', 'mixed', 'mixed', 'mixed', 'unjudged', 'unjudged', 'unjudged']);
  assert.equal(Math.floor(.3 * 250), MIXING_CODES.mixed); assert.equal(Math.floor(.1 * 250), MIXING_CODES.some);
});

test('one byte per cell (label - 1 order) is packed at texel `label` of 4096-wide rows; the rest reads not judged', () => {
  assert.equal(packMixing(null), null);
  const small = packMixing(Uint8Array.of(10, 80, 255));
  assert.deepEqual([small.rows, small.count, small.data.length], [1, 4, MIX_WIDTH]);
  assert.deepEqual([...small.data.slice(0, 6)], [255, 10, 80, 255, 255, 255]);
  const bytes = new Uint8Array(40513).map((_, i) => i % 250), big = packMixing(bytes);
  assert.deepEqual([big.rows, big.count], [Math.ceil(40514 / 4096), 40514]);
  for (const label of [1, 4095, 4096, 4097, 40513]) assert.equal(big.data[(label / MIX_WIDTH | 0) * MIX_WIDTH + label % MIX_WIDTH], bytes[label - 1], `label ${label}`);
  assert.equal(packMixing(new Uint8Array(4095)).rows, 1); assert.equal(packMixing(new Uint8Array(4096)).rows, 2);
});

test('the cross-fade: only a second or more after the previous result, 3 frames over 300 ms, a redraw between steps shows the step before', () => {
  assert.equal(TINT_MS, 300); assert.equal(TINT_FRAMES, 3); assert.equal(TINT_AFTER_MS, 1000);
  for (const since of [0, 100, 500, 999, Infinity, NaN]) assert.equal(tintPlan(since), null, `${since}`);
  for (const since of [1000, 1500, 60000]) assert.deepEqual(tintPlan(since), { duration: 300, frames: 3 });
  assert.equal(tintPlan(5000, true), null, 'reduced motion');
  const plan = tintPlan(2000);
  assert.deepEqual([0, 149, 150, 299, 300, 1e6].map(t => tintValue(plan, t)), [1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1]);
  assert.deepEqual([0, 10, 150, 299].map(t => tintNext(plan, t)), [150, 150, 300, 300]);
  assert.equal(tintNext(plan, 300), null); assert.equal(tintValue(null, 0), 1); assert.equal(tintNext(null, 0), null);
});

/** A WebGL2 context that records calls (name and arguments); constants are distinct numbers. */
function recordingGL(log) {
  const constants = new Map(), constant = key => { if (!constants.has(key)) constants.set(key, 1000 + constants.size); return constants.get(key); };
  return new Proxy({}, { get(_, key) {
    if (typeof key !== 'string') return undefined;
    const unit = /^TEXTURE(\d+)$/.exec(key);
    if (unit) return 33984 + Number(unit[1]);   // the real TEXTUREn values: TEXTURE0 + n is TEXTUREn
    if (/^[A-Z0-9_]+$/.test(key)) return constant(key);
    return (...args) => { log.push([key, ...args]); return key === 'createTexture' ? { texture: log.length } : key === 'getParameter' ? 4096 : key === 'isContextLost' ? false : {}; };
  } });
}
const SYNC = ['getError', 'getParameter', 'readPixels', 'checkFramebufferStatus', 'getShaderParameter', 'getProgramParameter', 'getUniformLocation', 'finish'];

test('every program built on the colour functions binds the mixing textures on their own units with the cluster lookup', () => {
  const log = [], gl = recordingGL(log), colours = new ColourGPU(gl);
  const uploads = log.filter(c => c[0] === 'texImage2D' && c[3] === gl.R8);
  assert.equal(uploads.length, 2, 'two R8 textures, empty to start (no result)');
  assert.deepEqual(uploads[0].slice(1, 10).map(v => typeof v === 'object' ? 'data' : v), [gl.TEXTURE_2D, 0, gl.R8, MIX_WIDTH, 1, 0, gl.RED, gl.UNSIGNED_BYTE, 'data']);
  assert.ok(log.some(c => c[0] === 'pixelStorei' && c[1] === gl.UNPACK_ALIGNMENT && c[2] === 1));
  assert.deepEqual(MIX_UNITS, [7, 8]);
  const mixingUnits = log.filter(c => c[0] === 'activeTexture').map(c => c[1] - 33984);
  assert.ok(mixingUnits.includes(7) && mixingUnits.every(unit => unit === 7 || unit === 0), `uploads only touch unit 7: ${mixingUnits}`);
  const u = { uMixFrom: 'from', uMixTo: 'to', uMixing: 'on', uMixT: 't', uMixCount: 'count', uClusterColor: 'cluster' };
  const bound = mode => { log.length = 0; colours.bind(u, { hover: 0 }, mode); return Object.fromEntries(log.filter(c => c[0].startsWith('uniform')).map(c => [c[1], c.slice(2)])); };
  assert.deepEqual(bound('mixing').on, [1]); assert.deepEqual(bound('mixing').cluster, [false]);
  assert.deepEqual(bound('cluster').on, [0]); assert.deepEqual(bound('cluster').cluster, [true]);
  assert.deepEqual(bound('identity').on, [0]); assert.deepEqual(bound('cell').on, [0], 'transcripts coloured by their cell');
  const b = bound('mixing');
  assert.deepEqual([b.from, b.to, b.t, b.count], [[7], [8], [1], [0, 0]]);
  assert.deepEqual(log.filter(c => c[0] === 'activeTexture').map(c => c[1] - 33984), [5, 6, 7, 8], 'cluster lookup and style on 5-6, mixing on 7-8');
  // A new result: shown at once, or kept as `from` to fade.
  colours.mixing.set(packMixing(Uint8Array.of(1, 2)));
  assert.deepEqual([colours.mixing.to, colours.mixing.t, colours.mixing.counts[1]], [1, 1, 3]);
  colours.mixing.set(packMixing(Uint8Array.of(1, 2, 3)), true);
  assert.deepEqual([colours.mixing.to, colours.mixing.t, colours.mixing.counts], [0, 0, [4, 3]], 'the one on screen became `from`');
  assert.deepEqual(bound('mixing').count, [3, 4]);
  assert.deepEqual(log.filter(c => SYNC.includes(c[0])), [], 'binding asks the GPU nothing');
});

test('transcripts take the cells\' colours in 2-D; in 3-D their own choice, where "Cell" follows the Mixing colours too', () => {
  const o = (cellColorBy, transcriptColorBy) => ({ cellColorBy, transcriptColorBy });
  assert.equal(pointColourBy(o('mixing', 'cell'), false), 'mixing'); assert.equal(pointColourBy(o('cluster', 'cell'), false), 'cluster');
  assert.equal(pointColourBy(o('mixing', 'cell'), true), 'mixing', '3-D: Show on the cells colours their transcripts too');
  assert.equal(pointColourBy(o('mixing', 'cluster'), true), 'cluster', 'unless they are coloured by cluster');
  assert.equal(pointColourBy(o('identity', 'cell'), true), 'cell', 'as before otherwise');
  assert.equal(pointColourBy(o('cluster', 'cell'), true), 'cell');
});

/** The Stage scheduler (wake/tick/plan) on a fake loop, as render-loop.test.mjs builds it, with draw() recording the
 *  cross-fade weight it would draw. */
function scheduler({ cellColorBy = 'mixing', reduced = false } = {}) {
  const loop = fakeLoop(), draws = [], uploads = [];
  const stage = Object.create(Stage.prototype);
  Object.assign(stage, {
    canvas: { width: 1000, height: 800 }, camera: { x: 0, y: 0, zoom: 1 }, listeners: new Set(), tween: null, raf: null, _dirty: false,
    orbit: { tick() {}, transition: null, active: false, enabled: false }, gl: {}, textures: {}, flash: null, flashDue: null,
    options: { trails: false, cells: true, hover: 0, selected: 0, cellColorBy, transcriptColorBy: 'cell' },
    mixing: { bytes: null, packed: null, at: -Infinity }, tint: null, tintDue: null, calmMotion: false, animators: new Set(),
    motionQuery: { matches: reduced },
    clusterGPU: { mixing: { t: 1, set(packed, fade) { uploads.push([packed?.count ?? 0, fade]); this.t = fade ? 0 : 1; } } },
  });
  stage.draw = function () { this.dirty = false; draws.push({ t: loop.now, mix: this.clusterGPU.mixing.t }); };
  const result = (t, bytes) => loop.at(t, () => { stage.setMixing(bytes, { at: t }); });
  return { loop, draws, uploads, stage, result };
}
const between = (draws, a, b) => draws.filter(d => d.t >= a && d.t < b);

test('a whole-sample run (a result a second): each cross-fades in 3 frames (1/3, 2/3, 1), then nothing is scheduled', () => {
  const h = scheduler();
  for (let k = 0; k < 5; k++) h.result(1000 + 1000 * k, new Uint8Array(4).fill(k * 30));
  h.loop.run(7000);
  assert.deepEqual(between(h.draws, 1000, 2000).map(d => d.mix), [1], 'the first result lands at once (nothing to fade from)');
  for (let k = 1; k < 5; k++) {
    const own = between(h.draws, 1000 + 1000 * k, 2000 + 1000 * k);
    assert.deepEqual(own.map(d => +d.mix.toFixed(3)), [.333, .667, 1], `result ${k}`);
    assert.deepEqual(own.map(d => Math.round((d.t - (1000 + 1000 * k)) / 50) * 50), [0, 150, 300], 'on the 150 ms steps');
  }
  assert.deepEqual(h.uploads.map(u => u[1]), [false, true, true, true, true]);
  assert.ok(h.loop.idle(), 'idle between and after results');
});

test('fast results land in their own frame; no fade without mixing colours, with reduced motion, with calm motion, or for the same bytes', () => {
  const fast = scheduler();
  for (let k = 0; k < 20; k++) fast.result(1000 + 100 * k, new Uint8Array(4).fill(k));
  fast.loop.run(4000);
  assert.equal(fast.draws.length, 20, 'one draw per result'); assert.ok(fast.draws.every(d => d.mix === 1));
  assert.ok(fast.loop.idle());

  for (const [name, h, expectDraws] of [['identity', scheduler({ cellColorBy: 'identity' }), 0], ['reduced', scheduler({ reduced: true }), 3]]) {
    for (let k = 0; k < 3; k++) h.result(1000 + 2000 * k, new Uint8Array(4).fill(k));
    h.loop.run(9000);
    assert.equal(h.draws.length, expectDraws, `${name}: ${h.draws.length} draws`);
    assert.ok(h.draws.every(d => d.mix === 1)); assert.ok(h.uploads.every(u => u[1] === false), `${name}: never faded`);
    assert.ok(h.loop.idle());
  }
  const calm = scheduler(); calm.stage.calmMotion = true;
  calm.result(1000, new Uint8Array(2)); calm.result(3000, Uint8Array.of(1, 1));
  calm.loop.run(5000);
  assert.deepEqual(calm.draws.map(d => d.mix), [1, 1]);
  // motion: false (paused, a late result for the frame on screen): at once, and the calm switch stays as it was.
  const still = scheduler();
  still.result(1000, new Uint8Array(2));
  still.loop.at(3000, () => { assert.equal(still.stage.setMixing(Uint8Array.of(2, 2), { at: 3000, motion: false }), null); });
  still.result(5000, Uint8Array.of(3, 3));
  still.loop.run(7000);
  assert.deepEqual(between(still.draws, 3000, 5000).map(d => d.mix), [1], 'lands in one frame');
  assert.deepEqual(between(still.draws, 5000, 7000).map(d => +d.mix.toFixed(3)), [.333, .667, 1], 'the next update fades again');
  assert.equal(still.stage.calmMotion, false);
  const same = scheduler(), bytes = Uint8Array.of(5, 6);
  same.result(1000, bytes); same.result(3000, bytes);
  same.loop.run(5000);
  assert.equal(same.draws.length, 1, 'the same result again uploads and draws nothing');
  assert.equal(same.stage.mixingCode(1), 5); assert.equal(same.stage.mixingCode(2), 6);
  assert.equal(same.stage.mixingCode(3), null); assert.equal(same.stage.mixingCode(0), null);
  same.loop.with(() => same.stage.setMixing(null));
  assert.equal(same.stage.mixingCode(1), null); assert.equal(same.stage.mixing.at, -Infinity, 'cleared: the next result does not fade');
});

test('animators get their frames in the stage\'s animation frames (no stage draw of their own), and nothing while idle', () => {
  const h = scheduler(), frames = [];
  const animator = { steps: [], due() { return this.steps[0] ?? null; }, frame(now) { frames.push(now); this.steps.shift(); } };
  h.loop.with(() => h.stage.addAnimator(animator));
  h.loop.at(1000, () => { animator.steps = [1000, 1150, 1300, 1450]; h.stage.replan(); });
  h.loop.run(3000);
  assert.deepEqual(frames.map(t => Math.round(t / 50) * 50), [1000, 1150, 1300, 1450], 'each frame at its step, on a vsync');
  assert.equal(h.draws.length, 0, 'the stage itself did not redraw');
  assert.ok(h.loop.idle());
  // With a cross-fade at the same time: the same frames (the overlay and the cells change together).
  h.result(5000, Uint8Array.of(1)); h.result(7000, Uint8Array.of(90));
  h.loop.at(7000, () => { animator.steps = [7000, 7150, 7300, 7450]; h.stage.replan(); });
  frames.length = 0; h.loop.run(9000);
  assert.deepEqual(between(h.draws, 7000, 9000).map(d => d.t), frames.slice(0, 3), 'the tint frames are animator frames too');
  assert.equal(frames.length, 4);
  assert.ok(h.loop.idle());
  h.stage.removeAnimator(animator); assert.equal(h.stage.animators.size, 0);
});

/** stage.js with its harness hook bound to `hook` (as harness-hook.test.mjs loads modules). */
async function hookedStage(hook) {
  globalThis.__qcStageHook = hook;
  const dataURL = s => `data:text/javascript;base64,${Buffer.from(s).toString('base64')}`;
  const source = (await readFile(new URL('stage.js', SRC), 'utf8')).replace(/from '\.\/([^']+)'/g, (_, name) =>
    `from ${JSON.stringify(name === 'harness-hook.mjs' ? dataURL('export const harness = globalThis.__qcStageHook;') : new URL(name, SRC).href)}`);
  return import(dataURL(`${source}\n// qc-stage ${Math.random()}`));
}

test('the harness counts the frames the mixing check\'s motion adds (not a result\'s own frame)', async () => {
  const counts = {}, hook = { count(name, n = 1) { counts[name] = (counts[name] || 0) + n; } };
  const { Stage: Hooked } = await hookedStage(hook);
  const h = scheduler();
  Object.setPrototypeOf(h.stage, Hooked.prototype);
  h.stage.draw = function () { this.dirty = false; h.draws.push({ t: h.loop.now, mix: this.clusterGPU.mixing.t }); };
  h.result(1000, Uint8Array.of(1)); h.result(3000, Uint8Array.of(90));
  h.loop.run(5000);
  assert.equal(h.draws.length, 4);
  assert.equal(counts.qcAnimFrames, 2, 'the cross-fade\'s two later steps');
  // An animator's frame that is not motion (frame() returns false: a refresh after an option changed) is not counted;
  // its motion frames are.
  const frames = [], animator = { steps: [], due() { return this.steps[0] ?? null; },
    frame(now) { frames.push(now); return this.steps.shift() === 'motion'; } };
  h.loop.with(() => h.stage.addAnimator(animator));
  h.loop.at(6000, () => { animator.steps = ['refresh']; animator.due = () => animator.steps.length ? 6000 : null; h.stage.replan(); });
  h.loop.run(6500);
  assert.equal(frames.length, 1); assert.equal(counts.qcAnimFrames, 2, 'the refresh is not motion');
  h.loop.at(7000, () => { animator.steps = ['motion']; animator.due = () => animator.steps.length ? 7000 : null; h.stage.replan(); });
  h.loop.run(7500);
  assert.equal(frames.length, 2); assert.equal(counts.qcAnimFrames, 3);
  assert.ok(h.loop.idle());
});

test('a restored context uploads the newest mixing result again; results and draws ask the GPU nothing', () => {
  const loop = fakeLoop(), log = [], listeners = {};
  const canvas = { width: 0, height: 0, addEventListener(type, fn) { listeners[type] = fn; }, getContext: () => recordingGL(log), getBoundingClientRect: () => ({ width: 400, height: 300 }) };
  const stage = loop.with(() => new Stage(canvas));
  assert.ok(stage.clusterGPU instanceof ColourGPU); assert.ok(stage.clusterGPU.mixing instanceof MixingTextures);
  loop.run(100);
  log.length = 0;
  const frame = { nx: 2, ny: 1, x0: 0, y0: 0, voxel: 1, iter: 1, labels: Uint32Array.of(1, 2), consensus: null, age: null, layers: [] };
  loop.with(() => { stage.set({ cellColorBy: 'mixing' }); stage.setFrame(frame); stage.setMixing(Uint8Array.of(80, 255)); });
  loop.run(400);
  const R8 = stage.gl.R8, packed = log.filter(c => c[0] === 'texImage2D' && c[3] === R8);
  assert.equal(packed.length, 1, 'one upload for the result'); assert.deepEqual([...packed[0][9].slice(0, 4)], [255, 80, 255, 255]);
  assert.deepEqual(log.filter(c => SYNC.includes(c[0])).map(c => c[0]), [], 'no synchronous call in setMixing or draw');
  assert.ok(log.some(c => c[0] === 'drawArrays'));
  loop.at(500, () => listeners.webglcontextlost({ preventDefault() {} }));
  loop.at(600, () => listeners.webglcontextrestored());
  log.length = 0;
  loop.run(700);
  const r8 = log.filter(c => c[0] === 'texImage2D' && c[4] === MIX_WIDTH && c[5] === 1);
  assert.ok(r8.length >= 3, 'two empty textures, then the newest result');
  assert.deepEqual([...r8.at(-1)[9].slice(0, 4)], [255, 80, 255, 255]);
  assert.equal(stage.clusterGPU.mixing.t, 1, 'shown at once'); assert.equal(stage.tint, null);
});
