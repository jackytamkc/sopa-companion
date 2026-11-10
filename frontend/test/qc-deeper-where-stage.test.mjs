import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { Stage, MASK_FS, XENIUM_FS, COLORS, SPILL_VS, SPILL_FS, SPILL_UM, SPILL_MIN_PX, SPILL_MAX_PX, SPILL_DIM, SPILL_LIT, SPILL_COLORS, GLOW_DOTS } from '../src/stage.js';
import { fakeLoop } from './fake-loop.mjs';

// Go deeper's "Show where on the image" on the stage (Stage.setSpill): the window's
// cell and sources stay lit while every other cell fades, the usual transcripts give way to the window's dots, in 2-D
// only; with it off nothing draws differently. Then the endpoint's call (api.js) and main.js's few lines between the
// window, the stage, the arrows and the camera.

/** A WebGL2 context that records calls (name and arguments); getParameter answers `point` for the point size range. */
function recordingGL(log, point = [1, 1023]) {
  const constants = new Map(), constant = key => { if (!constants.has(key)) constants.set(key, 1000 + constants.size); return constants.get(key); };
  return new Proxy({}, { get(_, key) {
    if (typeof key !== 'string') return undefined;
    if (/^TEXTURE(\d+)$/.test(key)) return 33984 + Number(key.slice(7));
    if (/^[A-Z0-9_]+$/.test(key)) return constant(key);
    return (...args) => { log.push([key, ...args]); return key === 'getParameter' ? (args[0] === constant('ALIASED_POINT_SIZE_RANGE') ? point : 4096) : {}; };
  } });
}
const SPILL_U = { uCanvas: 'canvas', uView: 'view', uSpillSize: 'size', 'uSpillColors[0]': 'colors', uSpillAlpha: 'alpha' };
/** A Stage with what draw() needs in 2-D, on a recording context; drawTranscripts() records that it ran. */
function spillStage(log, { active = false } = {}) {
  const stage = Object.create(Stage.prototype), drawn = [];
  Object.assign(stage, { gl: recordingGL(log), canvas: { width: 1000, height: 800 }, camera: { x: 50, y: 40, zoom: 10 }, pixelRatio: 2, image: null,
    orbit: { active, uniforms() {} }, textures: { channels: [], lab: {}, age: {} }, quadVao: 'quad', spillVao: 'spillVao', spillBuffer: 'spillBuffer',
    frame: { x0: 0, y0: 0, nx: 2, ny: 2, voxel: 1, iter: 3 }, limits: { point: 1023 }, drawCount: 0,
    options: { cells: true, mode: 0, fill: 0.42, trails: false, edgePx: 1.6, hover: 0, selected: 2, cellColorBy: 'identity' }, clusterGPU: { bind() {} },
    maskProgram: { program: 'mask', uniforms: { uFocusOn: 'focusOn', uSpillOn: 'spillOn', 'uSpillLit[0]': 'lit' } },
    spillProgram: { program: 'spill', uniforms: SPILL_U },
    drawTranscripts() { drawn.push('transcripts'); }, raf: 1, _dirty: false });
  return { stage, drawn };
}
const uniforms = log => Object.fromEntries(log.filter(c => c[0].startsWith('uniform') && typeof c[1] === 'string').map(c => [c[1], c.slice(2)]));
const dots = (...rows) => Float32Array.from(rows.flat());   // [x, y, colour slot, opacity] per dot

test('the mask: the fading is behind uSpillOn, lights the cell and its sources, and leaves every pixel as it was when off', () => {
  for (const shader of [MASK_FS, XENIUM_FS]) {
    assert.match(shader, /uniform bool uSpillOn;\nuniform uint uSpillLit\[6\];/);
    assert.ok(shader.includes(`bool spillLit(uint id) {\n  for (int k = 0; k < ${SPILL_LIT}; k++) if (id == uSpillLit[k]) return id != 0u;\n  return false;\n}`), 'never the background');
  }
  const main = MASK_FS.slice(MASK_FS.indexOf('void main()'));
  // Every use of the spill uniforms in main() is gated: with uSpillOn false each line is the line it was.
  const uses = main.split('\n').filter(line => /uSpill|spillLit/.test(line));
  assert.deepEqual(uses.map(l => l.trim().replace(/\s*\/\/.*$/, '')), [
    'bool dimmed = uSelected != 0u && id != uSelected && !uSpillOn;',
    `if (uSpillOn && !spillLit(id)) col.a *= ${SPILL_DIM};`]);
  // The fading comes after the trails (a cell changing next door cannot light up again) and before the clusters' weight.
  assert.ok(main.indexOf('if (uSpillOn && !spillLit(id))') > main.indexOf('if (uTrails && recent > 0.01)') && main.indexOf('if (uSpillOn && !spillLit(id))') < main.indexOf('col.a *= clusterWeight(id);'));
  // Outlines of the focus stay as they are; only cells left out of it fade (the cell and its sources keep their fill). The
  // focus's own greying is on col too, never the fill ("on col after the edge colour,
  // never on the fill"); it stacks with this fading for the cells neither lights.
  assert.doesNotMatch(main, /fill \*= .*(focus|spill)|col = vec4\(base, .*spill/i);
  assert.equal(COLORS.includes('uSpill'), false, 'not in the colour functions the transcripts and 3-D share');
  assert.equal(SPILL_DIM, 0.12); assert.ok(SPILL_DIM < 0.3 * 0.45, 'far below the selection\'s dim');
  assert.deepEqual([SPILL_UM, SPILL_MIN_PX, SPILL_MAX_PX, SPILL_LIT, SPILL_COLORS], [0.6, 4, 14, 6, 9]);
  // The dots: about 0.6 µm on the image, clamped in device px, a dark rim; the colour by the gene's slot, the opacity by
  // the dot's own times uSpillAlpha (stepping back while the suspects glow).
  assert.match(SPILL_VS, /vSize = clamp\(uSpillSize\.x \* uView\.z, uSpillSize\.y, uSpillSize\.z\);/);
  assert.match(SPILL_VS, /uniform vec3 uSpillColors\[9\];/); assert.match(SPILL_FS, /if \(alpha < \.02\) discard;/);
  assert.ok(SPILL_VS.includes(`vColor = vec4(uSpillColors[clamp(int(aPos.z + .5), 0, ${SPILL_COLORS - 1})], aPos.w * uSpillAlpha);`));
});

test('while the suspects glow (setGlow) the window\'s dots step back to GLOW_DOTS of their opacity; the stage redraws only when that changes', () => {
  // "The ordinary where dots stay, at lower opacity, so the glow leads." The glow repaints its own canvas; the
  // stage draws again only for the dots' opacity, when a glow comes or goes while the dots are on.
  const loop = fakeLoop(), log = [], { stage } = spillStage(log);
  const glow = key => ({ key, points: Float32Array.of(10, 20, 0.9), giver: null });
  loop.with(() => stage.setSpill({ key: 'k', cell: 2, lit: [3], points: dots([10, 20, 0, 1]), colors: ['#f0e442'] }));
  stage.draw(0); assert.deepEqual(uniforms(log).alpha, [1]);
  stage._dirty = false;
  assert.equal(loop.with(() => stage.setGlow(glow('a'))), true); assert.equal(stage.dirty, true, 'the dots step back: one redraw');
  log.length = 0; stage.draw(0); assert.deepEqual(uniforms(log).alpha, [GLOW_DOTS]); assert.equal(GLOW_DOTS, 0.5);
  stage._dirty = false;
  assert.equal(loop.with(() => stage.setGlow(glow('b'))), true); assert.equal(stage.dirty, false, 'another glow: the stage stays still');
  assert.equal(loop.with(() => stage.setGlow(glow('b'))), false);
  assert.equal(loop.with(() => stage.setGlow(null)), true); assert.equal(stage.dirty, true);
  log.length = 0; stage.draw(0); assert.deepEqual(uniforms(log).alpha, [1]);
  // Without the dots a glow never redraws the stage.
  loop.with(() => stage.setSpill(null)); stage._dirty = false;
  loop.with(() => stage.setGlow(glow('c'))); assert.equal(stage.dirty, false);
  loop.with(() => stage.setGlow(null)); assert.equal(stage.dirty, false);
});

test('setStartOutline: kept for the overlay to draw, the stage\'s own drawing untouched; the same points again: no work', () => {
  // "The cell's starting outline is drawn dashed white on the stage while Show is on" (drawn on the overlay by
  // qc-overlay.mjs, which reads stage.startOutline in the refresh frame this asks for: qc-overlay.test.mjs).
  const loop = fakeLoop(), log = [], { stage } = spillStage(log);
  let replans = 0;
  stage.replan = () => { replans++; };
  stage.draw(0);
  const before = log.map(c => [...c]);
  const ring = Float32Array.of(1, 1, 5, 1, 5, 4, 1, 4);
  assert.equal(stage.startOutline ?? null, null);
  assert.equal(loop.with(() => stage.setStartOutline({ points: ring, closed: true })), true); assert.equal(replans, 1, 'the overlay\'s refresh asked for');
  assert.deepEqual({ ...stage.startOutline, points: [...stage.startOutline.points] }, { points: [...ring], closed: true, count: 4 });
  assert.notEqual(stage.startOutline.points, ring, 'a copy: the caller\'s array may change');
  assert.equal(stage.dirty, false, 'the stage itself does not redraw for it');
  log.length = 0; stage.draw(0);
  assert.deepEqual(log, before, 'the same GL calls and uniforms as without it');
  assert.equal(stage.setStartOutline({ points: Float32Array.from(ring) }), false, 'the same points (closed by default): no work'); assert.equal(replans, 1);
  assert.equal(stage.setStartOutline({ points: ring, closed: false }), true, 'open'); assert.equal(stage.startOutline.closed, false);
  assert.equal(stage.setStartOutline({ points: Float32Array.of(1, 1, 5, 1, 5, 4) }), true, 'another outline'); assert.equal(stage.startOutline.count, 3);
  // An odd last value is dropped; fewer than two points, a point not finite or a plain array: none.
  assert.equal(stage.setStartOutline({ points: Float32Array.of(1, 1, 5, 1, 9) }), true); assert.equal(stage.startOutline.count, 2);
  for (const bad of [{ points: Float32Array.of(1, 1) }, { points: Float32Array.of(1, 1, NaN, 2) }, { points: [1, 1, 5, 1] }, {}, null]) {
    stage.setStartOutline({ points: ring });
    assert.equal(stage.setStartOutline(bad), true, JSON.stringify(bad)); assert.equal(stage.startOutline, null);
  }
  assert.equal(stage.setStartOutline(null), false, 'none again: no work');
});

test('setSpill: redraws only on a change, uploads the dots once per change; null draws exactly as before it', () => {
  const log = [], { stage, drawn } = spillStage(log);
  stage.draw(0);
  const before = log.map(c => c[0]), plain = uniforms(log);
  assert.deepEqual(plain.spillOn, [0]); assert.equal(plain.lit, undefined); assert.deepEqual(drawn, ['transcripts']);
  assert.ok(!log.some(c => c[0] === 'useProgram' && c[1] === 'spill'), 'the dots\' pass never runs');
  const spill = { key: 'r1:a', cell: 2, lit: [2, 3, 3, 0], points: dots([10, 20, 0, 1], [11, 21, 1, .5], [12, 22, 6, .35]), colors: ['#f0e442', '#44bb99', 'bad'] };
  log.length = 0;
  assert.equal(stage.setSpill(spill), true); assert.equal(stage.dirty, true);
  assert.deepEqual(log.filter(c => c[0] === 'bufferData').map(c => [...c[2]]), [[...spill.points]], 'uploaded once');
  assert.ok(log.some(c => c[0] === 'vertexAttribPointer' && c[1] === 0 && c[2] === 4 && c[5] === 16), 'x, y, slot, opacity: 16 bytes a dot');
  assert.deepEqual([...stage.spill.lit], [2, 3, 0, 0, 0, 0], 'the cell first, each once, never the background');
  assert.deepEqual([...stage.spill.colors.slice(0, 9)].map(v => +v.toFixed(3)), [0.941, 0.894, 0.259, 0.267, 0.733, 0.6, 0.6, 0.64, 0.7]);
  stage._dirty = false; log.length = 0;
  assert.equal(stage.setSpill({ ...spill }), false, 'the same answer: no redraw, no upload'); assert.equal(stage.dirty, false); assert.equal(log.length, 0);
  assert.equal(stage.setSpill({ ...spill, key: 'r1:b' }), true, 'another answer');
  // Drawn: the mask lights the cell and its source, the usual transcripts give way to the dots, over the cells.
  log.length = 0; drawn.length = 0; stage.draw(0);
  const u = uniforms(log);
  assert.deepEqual(u.spillOn, [1]); assert.deepEqual([...u.lit[0]], [2, 3, 0, 0, 0, 0]); assert.deepEqual(drawn, [], 'the usual transcripts are hidden');
  const names = log.map(c => c[0]), mask = log.findIndex(c => c[0] === 'useProgram' && c[1] === 'mask'), pass = log.findIndex(c => c[0] === 'useProgram' && c[1] === 'spill');
  assert.ok(mask >= 0 && pass > mask, 'the dots after the cells');
  assert.deepEqual(log.find((c, i) => i > pass && c[0] === 'drawArrays').slice(2), [0, 3]);
  assert.deepEqual(u.size, [SPILL_UM, SPILL_MIN_PX * 2, SPILL_MAX_PX * 2], 'never under 4 nor over 14 CSS px (device px at 2x)');
  assert.deepEqual(u.view, [50, 40, 10]); assert.equal(u.colors[0].length, 3 * SPILL_COLORS);
  assert.ok(names.lastIndexOf('bindVertexArray') > pass && log[names.lastIndexOf('bindVertexArray')][1] === 'quad', 'the quad bound back');
  // Off: exactly the calls of before (nothing left behind).
  assert.equal(stage.setSpill(null), true); log.length = 0; drawn.length = 0; stage.draw(0);
  assert.deepEqual(log.map(c => c[0]), before); assert.deepEqual(uniforms(log), plain); assert.deepEqual(drawn, ['transcripts']);
  assert.equal(stage.setSpill(null), false); assert.equal(stage.setSpill({ cell: 0, points: dots() }), false, 'no cell: none');
  assert.equal(stage.setSpill({ cell: 2, points: [1, 2, 0, 1] }), false, 'dots must be a Float32Array');
});

test('the dots\' sizes stay under the GPU\'s largest point; none to draw keeps the transcripts hidden; 3-D draws as without it', () => {
  const log = [], { stage, drawn } = spillStage(log);
  stage.limits.point = 20; stage.setSpill({ key: 'k', cell: 2, lit: [3], points: dots([1, 1, 0, 1]), colors: ['#f0e442'] });
  stage.draw(0);
  assert.deepEqual(uniforms(log).size, [SPILL_UM, 8, 20]);
  stage.setSpill({ key: 'empty', cell: 2, lit: [3], points: dots(), colors: [] });
  log.length = 0; drawn.length = 0; stage.draw(0);
  assert.deepEqual(uniforms(log).spillOn, [1], 'the fading still shows'); assert.deepEqual(drawn, []);
  assert.ok(!log.some(c => c[0] === 'useProgram' && c[1] === 'spill'), 'no dots, no pass');
  // 3-D: no fading, no dots, the transcripts as usual.
  const log3 = [], three = spillStage(log3, { active: true });
  three.stage.setSpill({ key: 'k', cell: 2, lit: [3], points: dots([1, 1, 0, 1]), colors: ['#f0e442'] });
  log3.length = 0; three.stage.draw(0);
  assert.deepEqual(uniforms(log3).spillOn, [0]); assert.deepEqual(three.drawn, ['transcripts']);
  assert.ok(!log3.some(c => c[0] === 'useProgram' && c[1] === 'spill'));
});

// The real Stage on a fake loop and a GL that records every call (gpu-sync.test.mjs's): no synchronous call while the dots
// change or draw, and a restored context uploads them again.
const SYNC = ['getError', 'getParameter', 'readPixels', 'checkFramebufferStatus', 'getShaderParameter', 'getProgramParameter', 'getActiveUniform',
  'getUniformLocation', 'getExtension', 'getShaderInfoLog', 'getProgramInfoLog', 'finish', 'getBufferSubData', 'clientWaitSync'];
function fakeGL(log) {
  const constants = new Map(), constant = key => { if (!constants.has(key)) constants.set(key, 1000 + constants.size); return constants.get(key); };
  return new Proxy({}, { get(_, key) {
    if (typeof key !== 'string') return undefined;
    if (/^[A-Z0-9_]+$/.test(key)) return constant(key);
    return (...args) => {
      log.push([key, ...args]);
      if (key === 'getParameter') return args[0] === constant('ALIASED_POINT_SIZE_RANGE') ? new Float32Array([1, 63]) : 4096;
      if (key === 'getShaderParameter' || key === 'getProgramParameter') return args[1] === constant('ACTIVE_UNIFORMS') ? 0 : true;
      if (key === 'getExtension' || key === 'isContextLost') return key === 'getExtension' ? null : false;
      return { id: log.length };
    };
  } });
}
test('the dots ask the GPU nothing while they change or draw; a restored context uploads them again; init reads the largest point', () => {
  const loop = fakeLoop(), log = [], listeners = {};
  const canvas = { width: 0, height: 0, addEventListener(type, fn) { listeners[type] = fn; }, getContext: () => fakeGL(log), getBoundingClientRect: () => ({ width: 1000, height: 800 }) };
  const stage = loop.with(() => new Stage(canvas));
  assert.equal(stage.limits.point, 63, 'ALIASED_POINT_SIZE_RANGE, once at init');
  assert.ok(stage.spillProgram && stage.spillVao && stage.spillBuffer);
  log.length = 0;
  const points = dots([1, 2, 0, 1], [3, 4, 1, .5]);
  loop.with(() => {
    stage.setFrame({ nx: 2, ny: 2, x0: 0, y0: 0, voxel: 1, iter: 1, labels: new Uint32Array(4), consensus: null, age: null, layers: [] });
    stage.setSpill({ key: 'a', cell: 1, lit: [2], points, colors: ['#f0e442', '#44bb99'] });
  });
  loop.run(500);
  const sync = log.filter(c => SYNC.includes(c[0])).map(c => c[0]);
  assert.deepEqual(sync, []); assert.ok(log.some(c => c[0] === 'drawArrays'), 'drawn');
  assert.equal(log.filter(c => c[0] === 'bufferData' && c[2] === points).length, 1);
  log.length = 0;
  loop.at(1000, () => listeners.webglcontextlost({ preventDefault() {} })); loop.run(1100);
  loop.at(1200, () => listeners.webglcontextrestored()); loop.run(1500);
  assert.equal(log.filter(c => c[0] === 'bufferData' && c[2] === points).length, 1, 'the dots uploaded again');
  assert.equal(stage.spill.count, 2);
});

test('api.qcDeeperWhere: its endpoint, its names encoded, quiet like every /qc request', async () => {
  const requests = [];
  const context = vm.createContext({ window: { fetch() {}, dispatchEvent() {} },
    fetch: async (url, options) => { requests.push({ url, ...options }); return { ok: true, status: 200, json: async () => ({ v: 1 }) }; } });
  const api = vm.runInContext(readFileSync(new URL('../src/api.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '').replaceAll('export ', '') + '\napi', context);
  assert.deepEqual(await api.qcDeeperWhere('run a', { seq: 12, cell: 1, sources: [2, 9], genes: ['GXPN', 'GYK-NNNN', 'A,B'], own: ['GYHNN'] }), { v: 1 });
  await api.qcDeeperWhere('run a', { seq: 'start', cell: 4, sources: [5] });
  assert.deepEqual(requests.map(r => [r.method, r.url]), [
    ['GET', '/api/v1/runs/run%20a/qc/deeper/where?seq=12&cell=1&sources=2,9&genes=GXPN,GYK-NNNN,A%2CB&own=GYHNN'],
    ['GET', '/api/v1/runs/run%20a/qc/deeper/where?seq=start&cell=4&sources=5&genes=&own=']]);
  for (const r of requests) assert.match(r.url, /\/qc(?:[/?]|$)/, 'quiet: the window says what went wrong in its own words');
});

test('main.js: the window\'s dots reach the stage and set the arrows aside; its one camera move is the stage\'s fit', () => {
  const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  assert.match(source, /onSpill: spill => deeperSpill\(spill\), onFly: \(rect, animate, view\) => deeperFly\(rect, animate, view\),/);
  const part = source.slice(source.indexOf('function deeperCentre(label)'), source.indexOf('function updateActiveCells()'));
  const calls = [], marks = new Map();
  const context = vm.createContext({
    state: { mode: 'run', run: { mode: 'tiled' }, selected: 2, current: 0, frames: [] },
    tiled: { overview: true }, cellStats: { current: null }, analysis: { view: { ringCells() {} } }, qcBoxes: new Map(), deeper: { root: {} },
    stage: { setSpill: s => calls.push(['setSpill', s]), fit: (...a) => calls.push(['fit', ...a]), setFocus() {}, set() {} },
    deeperArrows: { set: (f, o) => { calls.push(['arrows', f && f.cell]); return true; } },
    // drawOverlay's own first line lets go of a wish to draw it (asserted below on the source)
    drawOverlay: () => { calls.push(['overlay']); context.overlayWanted = false; }, updateCellCard() {}, placeLegend() {},
    paint: { mark: (key, fn) => { calls.push(['mark', key]); marks.set(key, fn); } },
  });
  vm.runInContext(`${part}\nthis.api = { deeperFocus, deeperSpill, deeperFly };`, context);
  const { deeperFocus, deeperSpill, deeperFly } = context.api;
  // P5: the arrows changed by the window's apply (focus.apply: a result painted) mark the overlay for the page's next
  // paint batch instead of drawing it there; the batch draws it once, and not at all when something drew it first.
  deeperFocus({ cell: 2, sources: [{ label: 3, share: 0.4 }], same: [], hot: 0, apply: true });
  assert.deepEqual(calls.splice(0), [['arrows', 2], ['mark', 'overlay']], 'the arrows as before, drawn with the next batch');
  marks.get('overlay')(); marks.get('overlay')();
  assert.deepEqual(calls.splice(0), [['overlay']], 'drawn once');
  deeperFocus({ cell: 2, sources: [{ label: 4, share: 0.4 }], same: [], hot: 0, apply: true });
  context.drawOverlay(); calls.splice(0); marks.get('overlay')();
  assert.deepEqual(calls.splice(0), [], 'drawn first by something else: never twice');
  assert.match(source, /function drawOverlay\(view = true\) \{\n  overlayWanted = false;/);
  // Any other change of the arrows (a row or an arrow under the pointer, a pin) is drawn at once, as is Show on the
  // image setting them aside and bringing them back.
  deeperFocus({ cell: 2, sources: [{ label: 4, share: 0.4 }], same: [], hot: 4 });
  assert.deepEqual(calls.splice(0), [['arrows', 2], ['overlay']], 'a hover: drawn at once');
  const spill = { key: 'k', cell: 2, lit: [3], points: new Float32Array(4), colors: [] };
  deeperSpill(spill);
  assert.deepEqual(calls.splice(0), [['setSpill', spill], ['arrows', null], ['overlay']], 'the dots on: the arrows set aside');
  deeperSpill({ ...spill, key: 'k2' });
  assert.deepEqual(calls.splice(0).map(c => c[0]), ['setSpill'], 'a new answer: the stage only');
  deeperSpill(null);
  assert.deepEqual(calls.splice(0), [['setSpill', null], ['arrows', 2], ['overlay']], 'off: the arrows back');
  deeperFly({ x0: 1, y0: 2, w: 3, h: 4 }, false);
  assert.deepEqual(calls.splice(0), [['fit', { x0: 1, y0: 2, w: 3, h: 4 }, false, 0, null]]);
  const view = { box: [0, 0, 120, 60], free: [440, 0, 1000, 800] };
  deeperFly({ x0: 1, y0: 2, w: 3, h: 4 }, true, view);
  assert.deepEqual(JSON.parse(JSON.stringify(calls.splice(0))), [['fit', { x0: 1, y0: 2, w: 3, h: 4 }, true, 0, view]]); assert.equal(context.tiled.overview, false, 'a whole sample leaves its overview');
  context.state.mode = 'pick'; deeperFly({ x0: 1, y0: 2, w: 3, h: 4 }, true);
  assert.deepEqual(calls, [], 'no run: no move');
});

test('main.js with the page\'s own paint batch: while a snapshot on its way holds it, a hover\'s arrows are drawn at once and only the window\'s apply waits for the snapshot', () => {
  // PaintBatch holds a mark that is not urgent until the snapshot it waits for is uploaded (release(token, true)), up to
  // HOLD_MS: the arrows of a result the window paints land with it, a row or an arrow under the pointer never waits.
  const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const part = source.slice(source.indexOf('function deeperCentre(label)'), source.indexOf('function updateActiveCells()'));
  const batch = readFileSync(new URL('../src/paint-batch.mjs', import.meta.url), 'utf8').replaceAll('export ', '');
  const calls = [], frames = [];
  let t = 0;
  const context = vm.createContext({
    performance: { now: () => t }, requestAnimationFrame: fn => { frames.push(fn); return frames.length; }, cancelAnimationFrame() {},
    setTimeout: () => 1, clearTimeout() {}, console,
    state: { mode: 'run', run: { mode: 'dense' }, selected: 2, current: 0, frames: [] },
    tiled: { overview: false }, cellStats: { current: null }, analysis: { view: { ringCells() {} } }, qcBoxes: new Map(), deeper: { root: {} },
    stage: { setSpill() {}, fit() {}, setFocus: f => calls.push(['setFocus', f && f.hot]), set() {} },
    deeperArrows: { set: f => { calls.push(['arrows', f && f.hot]); return true; } },
    drawOverlay: () => { calls.push(['overlay']); context.overlayWanted = false; }, updateCellCard() {}, placeLegend() {},
  });
  vm.runInContext(`${batch}\nvar paint = new PaintBatch();\nvar overlayWanted = false;\n${part}\nthis.api = { deeperFocus, deeperSpill, paint };`, context);
  const { deeperFocus, deeperSpill, paint } = context.api;
  const frame = () => { while (frames.length) frames.shift()(t); };
  const focus = (hot, apply = false) => ({ cell: 2, sources: [{ label: 3, share: 0.4 }, { label: 4, share: 0.2 }], same: [], hot, apply });
  deeperFocus(focus(0, true)); frame(); calls.splice(0);
  const token = paint.hold();   // the run plays: a snapshot is on its way
  t = 10; deeperFocus(focus(3)); frame();
  assert.deepEqual(calls.splice(0), [['setFocus', 3], ['arrows', 3], ['overlay']], 'the pointer enters row 3: its arrow lit at once, with its outline');
  t = 20; deeperSpill({ key: 'k', cell: 2, lit: [3], points: new Float32Array(4), colors: [] }); frame();
  assert.deepEqual(calls.splice(0), [['arrows', null], ['overlay']], 'Show on the image: the arrows set aside at once');
  deeperSpill(null); frame(); calls.splice(0);
  t = 30; deeperFocus(focus(3, true)); deeperFocus({ ...focus(3, true), sources: [{ label: 3, share: 0.5 }] }); frame();
  assert.deepEqual(calls.splice(0), [['setFocus', 3], ['arrows', 3], ['setFocus', 3], ['arrows', 3]], "the window's applies: their arrows wait");
  t = 400; paint.release(token, true);   // the snapshot lands
  assert.deepEqual(calls.splice(0), [['overlay']], 'drawn once, with the snapshot');
  frame(); assert.deepEqual(calls, []);
});
