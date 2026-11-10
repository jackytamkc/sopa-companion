import test from 'node:test';
import assert from 'node:assert/strict';
import { PriorLayers, labelOutline, COUNTERS } from '../src/prior-layers.js';
import { layerPath, LOD_PX, releaseTiles, FILL_SHARE, TILE_BUDGET } from '../src/prior-paths.mjs';
import { MAX_SHOWN } from '../src/seg-layers.mjs';
import { polygonSetFromLegacy, gridIndex } from '../src/polygons.mjs';
import { decodeVectors, VECTORS_QUERY } from '../src/vectors-decode.mjs';
import { encodeVectors } from './vectors-fixture.mjs';
import { readFile } from 'node:fs/promises';

// PriorLayers against recording stand-ins: a Path2D and a 2-D context that log every call, a WebGL context that logs
// its calls, and a clock and timers the test drives. What the old code drew is spelled out from its source: one
// setTransform(z, 0, 0, z, W/2 - x z, H/2 - y z), then per layer (Prior shapes, then Prior nuclei) strokeStyle, globalAlpha,
// lineWidth = width * ratio / z and one stroke of every polygon as moveTo, lineTo..., closePath, then the prior raster.

/** A Path2D that records its ops; addPath appends the other path's ops; `closes` notes how many subpaths each closePath
 *  closed a path holding (1: a one-polygon path). */
class RecPath {
  constructor() { this.ops = []; this.closes = []; RecPath.made.push(this); }
  moveTo(x, y) { this.ops.push(['M', x, y]); }
  lineTo(x, y) { this.ops.push(['L', x, y]); }
  closePath() { this.closes.push(this.ops.filter(op => op[0] === 'M').length); this.ops.push(['Z']); }
  addPath(path) { this.ops.push(...path.ops); }
}
RecPath.made = [];

test('range classes rebuild only changed grid tiles and unknown ids stay visible',()=>{
  const set=polygonSetFromLegacy(grid(60,60).map((p,i)=>({...p,id:i+1}))),box=[-100,-100,2000,2000];
  const draw=classes=>layerPath(set,box,1,RecPath,false,TILE_BUDGET,classes);
  const all=new Uint8Array(set.count);draw(all);RecPath.made=[];
  const hidden=all.slice();hidden[0]=1;const result=draw(hidden),index=gridIndex(set);
  assert.equal(result.polygons,set.count-1);
  assert.ok(RecPath.made.length<set.count/4,'only one changed grid cell and the union path rebuilt');
  const same=hidden.slice();RecPath.made=[];draw(same);assert.equal(RecPath.made.length,1);
  assert.equal(draw(new Uint8Array([1])).polygons,set.count-1,'a short array does not hide unmeasured cells');
  assert.ok(index.starts.length>2);
});

test('classes hide outlines completely, with no grey fill or stroke; set mismatch keeps every cell',()=>{
  const polys=grid(2,1).map((p,i)=>({...p,id:i+1})),h=setup({polys,camera:{x:20,y:10,zoom:2}});
  h.priors.vectors={...h.priors.vectors,nuclei:null,set:'joined'};h.priors.settings.xenium.on=true;
  h.priors.classes={gen:1,set:'joined',array:new Uint8Array([1,0])};let result=h.draw();
  const strokes=result.calls.filter(c=>c[0]==='stroke');assert.equal(strokes.length,1);
  assert.equal(subpaths(strokes[0][1][0]).length,1);assert.equal(result.calls.some(c=>c[0]==='fill'),false);
  h.priors.classes={gen:2,set:'old',array:new Uint8Array([1,1])};result=h.draw();
  assert.equal(subpaths(result.calls.find(c=>c[0]==='stroke')[1][0]).length,2);
});

test('worker class generations install progressively while held, and are sent once per change',()=>{
  const h=setup({info:{kind:'stain'},worker:true});h.priors.settings.xenium.on=true;
  h.priors.classes={gen:1,set:'s',array:new Uint8Array([0])};h.draw();const worker=h.w.workers[0];
  h.priors.classes={gen:2,set:'s',array:new Uint8Array([1])};h.draw();
  const first=bitmap();worker.reply(first);h.draw();
  assert.equal(h.priors.bitmap,first,'generation 1 installs while generation 2 is wanted');
  assert.equal(worker.requests.length,2);assert.equal(worker.requests[1].classes.gen,2);
  h.priors.classes={gen:3,set:'s',array:new Uint8Array([0])};
  const second=bitmap();worker.reply(second);h.draw();assert.equal(h.priors.bitmap,second);assert.equal(first.closed,1);
  const third=bitmap();worker.reply(third);h.draw();h.draw();assert.equal(h.priors.bitmap,third);assert.equal(worker.requests.length,3);
  h.stage.camera.x++;h.draw();assert.equal('classes' in worker.requests.at(-1),false,'camera moves do not recopy classes');
  h.priors.classes={gen:2,set:'s',array:new Uint8Array([1])};assert.equal(h.priors.classes.gen,3,'a late producer cannot reverse the generation');
  h.priors.ready={...h.priors.shown,classEpoch:h.priors.shown.classEpoch-1,bitmap:bitmap()};
  const stale=h.priors.ready.bitmap;h.draw();assert.equal(stale.closed,1);assert.equal(h.priors.bitmap,third);
});
/** Subpaths of a recorded path: [[x0, y0, x1, y1, ...], ...] (each one closed, or the test fails). */
function subpaths(path) {
  const out = [];
  for (const op of path.ops) {
    if (op[0] === 'M') out.push([op[1], op[2]]);
    else if (op[0] === 'L') out.at(-1).push(op[1], op[2]);
    else out.at(-1).closed = true;
  }
  for (const s of out) assert.ok(s.closed, 'every subpath is closed');
  return out;
}
/** A 2-D canvas whose context records calls with the style in force at each one (and `tag`: which canvas). */
function recCanvas(calls, tag) {
  const style = { strokeStyle: '#000000', globalAlpha: 1, lineWidth: 1, imageSmoothingEnabled: true };
  const g = new Proxy(style, {
    get: (t, k) => k in t ? t[k] : (...args) => { calls.push([k, args, { ...style }, tag]); },
    set: (t, k, v) => { t[k] = v; return true; },
  });
  return { width: 0, height: 0, getContext: () => g };
}
/** A WebGL context that logs its calls; `onCall(name)` runs on each (the tests charge an upload's time to the clock). */
function fakeGL(log, onCall = () => {}) {
  const ids = new Map(), constant = key => { if (!ids.has(key)) ids.set(key, 1000 + ids.size); return ids.get(key); };
  return new Proxy({}, { get: (_, key) => typeof key !== 'string' ? undefined : /^[A-Z0-9_]+$/.test(key) ? constant(key) : (...args) => {
    log.push([key, args]); onCall(key);
    return key === 'getShaderParameter' || key === 'getProgramParameter' ? true : { id: log.length };
  } });
}
/** Runs fn with document, Path2D, performance and the timers replaced; returns the world it ran in. */
function world(worker = false, { postMs = 0 } = {}) {
  const w = { now: 0, timers: [], calls: [], measures: [], cleared: [], canvases: [], workers: [] };
  const globals = {
    // The canvases in the order made: 0 the run's own raster, 1 the segmentation layers' (PriorLayers' constructor).
    document: { createElement: () => { const c = recCanvas(w.calls, w.canvases.length); w.canvases.push(c); return c; } },
    Path2D: RecPath,
    performance: { now: () => w.now, measure: (name, options) => w.measures.push({ name, ...options }), clearMeasures: name => w.cleared.push(name) },
    setTimeout: (fn, ms) => { w.timers.push({ fn, at: w.now + ms }); return w.timers.length; },
    clearTimeout: id => { if (w.timers[id - 1]) w.timers[id - 1].cleared = true; },
  };
  if (worker) Object.assign(globals, {
    Worker: class {
      constructor(url, options) { this.url = url; this.options = options; this.requests = []; w.workers.push(this); }
      postMessage(request, transfer) { assert.equal(transfer, undefined, 'the app keeps ownership of its vectors'); this.requests.push(structuredClone(request)); w.now += postMs; }
      terminate() { this.terminated = true; }
      reply(bitmap, fields = {}) {
        w.with(() => this.onmessage({ data: { id: this.requests.at(-1).id, bitmap, polygons: 1000, layer: 'xenium+nuclei', paintMs: 25, cost: 240, ...fields } }));
      }
    },
    OffscreenCanvas: class {},
    createImageBitmap: () => { throw Error('rasterisation must happen in the worker'); },
  });
  w.with = fn => {
    const saved = Object.fromEntries(Object.keys(globals).map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
    for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
    try { return fn(); } finally { for (const [k, d] of Object.entries(saved)) { if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k]; } }
  };
  /** Advance the clock, firing the timers that fall due. */
  w.advance = ms => w.with(() => { w.now += ms; for (const t of w.timers) if (!t.cleared && !t.fired && t.at <= w.now) { t.fired = true; t.fn(); } });
  return w;
}
/** `uploadMs`: what each texture upload costs on the test clock (the user's desktop: up to ~250 ms in the picker). */
function setup({ W = 800, H = 600, camera = { x: 500, y: 500, zoom: 2 }, info = { cells: 1 }, polys = grid(30, 30), uploadMs = 0, worker = false, postMs = 0, Layers = PriorLayers } = {}) {
  const w = world(worker, { postMs }), log = [];
  const charge = key => { if (key === 'texImage2D') w.now += uploadMs; };
  const stage = { canvas: { width: W, height: H }, camera: { ...camera }, ratio: 1, info, orbit: { active: false }, gl: fakeGL(log, charge), imageProgram: {}, movingUntil: 0, tween: null, dirty: false, frame: null };
  const priors = w.with(() => new Layers(stage, () => 'rgb(1,2,3)'));
  w.with(() => { priors.reset(); priors.vectors = { cells: polys, nuclei: nucleiOf(polys), source: { cells: 'xenium_cell_boundaries' } }; });
  stage.dirty = false;
  const draw = () => { w.calls.length = 0; log.length = 0; RecPath.made.length = 0; w.with(() => priors.draw()); return { calls: [...w.calls], gl: [...log] }; };
  return { w, stage, priors, draw, log, charge };
}
/** Abutting square cells on an n x m grid of 20 µm (shared edges, as Xenium's neighbouring cells have), ids as strings. */
function grid(n, m, size = 20) {
  const out = [];
  for (let j = 0; j < m; j++) for (let i = 0; i < n; i++) {
    const x = i * size + .25, y = j * size + .5;
    out.push({ id: `c${i}-${j}`, poly: [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]] });
  }
  return out;
}
/** A nucleus inside every 7th cell. */
const nucleiOf = polys => polys.filter((_, k) => k % 7 === 0).map(p => ({ id: p.id, poly: shrink(p.poly) }));
const shrink = poly => poly.map(([x, y]) => [x + 3, y + 3]).map(([x, y], k) => k === 1 || k === 2 ? [x - 6, y] : [x, y]).map(([x, y], k) => k >= 2 && k <= 3 ? [x, y - 6] : [x, y]);
const strokes = calls => calls.filter(c => c[0] === 'stroke');
const named = (log, name) => log.filter(c => c[0] === name);
const XENIUM = { on: true, color: '#8ecae6', opacity: .7, width: 1 }, NUCLEI = { on: true, color: '#ffd18a', opacity: 1, width: 2.2 };

test('each layer is one stroke of one path; closePath is only ever called on a one-polygon path (no giant path)', () => {
  const h = setup();
  Object.assign(h.priors.settings.xenium, XENIUM); Object.assign(h.priors.settings.nuclei, NUCLEI);
  const { calls } = h.draw();
  const s = strokes(calls);
  assert.equal(s.length, 2, 'one stroke per layer');
  assert.ok(s.every(c => c[1][0] instanceof RecPath), 'of a Path2D');
  assert.deepEqual(calls.filter(c => ['beginPath', 'moveTo', 'lineTo', 'closePath'].includes(c[0])), [], 'the context\'s own path is never built');
  const closes = RecPath.made.flatMap(p => p.closes);
  assert.ok(closes.length >= 40 && closes.every(n => n === 1), 'every closePath closes a path holding a single polygon');
  const tiles = RecPath.made.filter(p => !p.closes.length && p.ops.length && !s.some(c => c[1][0] === p));
  assert.ok(tiles.length > 1, 'polygons are appended to their cell\'s tile path');
});

test('the stroked geometry is the old path\'s for every polygon that can touch the canvas, with its transform and style', () => {
  const polys = grid(60, 50), h = setup({ polys, camera: { x: 610.3, y: 480.7, zoom: 1.7 } });
  Object.assign(h.priors.settings.xenium, XENIUM); Object.assign(h.priors.settings.nuclei, NUCLEI);
  h.priors.prior = recCanvas([]); h.priors.meta = { x0: 10, y0: 20, nx: 4, ny: 3, voxel: 2 };
  Object.assign(h.priors.settings.prior, { on: true, opacity: .65 });
  const { calls } = h.draw(), { camera: c, canvas: { width: W, height: H } } = h.stage, z = c.zoom;
  assert.deepEqual(calls.filter(k => k[0] === 'setTransform').map(k => k[1]), [[z, 0, 0, z, W / 2 - c.x * z, H / 2 - c.y * z]]);
  const [cells, nuclei] = strokes(calls);
  for (const [stroke, o, layer] of [[cells, XENIUM, polys], [nuclei, NUCLEI, nucleiOf(polys)]]) {
    assert.deepEqual([stroke[2].strokeStyle, stroke[2].globalAlpha, stroke[2].lineWidth], [o.color, o.opacity, o.width * 1 / z]);
    // The old path: every polygon of the layer, in order, as moveTo, lineTo..., closePath, at float32 precision.
    const old = layer.map(({ poly }) => poly.flat().map(Math.fround)), drawn = subpaths(stroke[1][0]);
    const key = s => s.join(','), want = new Set(old.map(key));
    assert.ok(drawn.every(s => want.has(key(s))), 'only the old subpaths, vertex for vertex');
    assert.equal(new Set(drawn.map(key)).size, drawn.length, 'each polygon once');
    // Everything whose stroke can reach a canvas pixel is there.
    const reach = (o.width / 2 + 1) / z, view = [c.x - W / 2 / z - reach, c.y - H / 2 / z - reach, c.x + W / 2 / z + reach, c.y + H / 2 / z + reach];
    const got = new Set(drawn.map(key));
    for (const { poly } of layer) {
      const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
      if (Math.max(...xs) >= view[0] && Math.min(...xs) <= view[2] && Math.max(...ys) >= view[1] && Math.min(...ys) <= view[3])
        assert.ok(got.has(key(poly.flat().map(Math.fround))), `polygon at ${poly[0]} is stroked`);
    }
    assert.ok(drawn.length < layer.length, 'polygons far off the canvas are not');
  }
  const image = calls.find(k => k[0] === 'drawImage');
  assert.deepEqual(image[1].slice(1), [10, 20, 8, 6]); assert.equal(image[2].globalAlpha, .65); assert.equal(image[2].imageSmoothingEnabled, false);
  assert.ok(calls.indexOf(image) > calls.indexOf(nuclei), 'the prior raster goes over the outlines, as before');
});

test('zoomed far out, tiles of sub-pixel polygons are stroked as their boxes', () => {
  const set = polygonSetFromLegacy(grid(40, 40, 10)), index = gridIndex(set);
  const near = 2 * LOD_PX / 10, far = LOD_PX / 10 / 2;
  const a = layerPath(set, [-1e4, -1e4, 1e4, 1e4], near, RecPath);
  assert.deepEqual([a.polygons, a.boxes], [1600, 0]);
  const b = layerPath(set, [-1e4, -1e4, 1e4, 1e4], far, RecPath);
  assert.deepEqual([b.polygons, b.boxes], [1600, 1600]);
  const rects = subpaths(b.path);
  assert.equal(rects.length, 1600);
  for (const r of rects) {
    assert.equal(r.length, 8, 'four corners');
    const [x0, y0, x1, , , y1] = r;
    assert.deepEqual([...r], [x0, y0, x1, y0, x1, y1, x0, y1]);
    assert.ok(Math.abs(x1 - x0 - 10) < 1e-3 && Math.abs(y1 - y0 - 10) < 1e-3, 'the polygon\'s own box');
  }
  assert.ok(index.maxSide.some(v => v > 0));
});

test('an expensive drawing is moved with the camera (linearly sampled) until the stage settles, then drawn once, exactly at the new camera', () => {
  const h = setup({ uploadMs: 120 });                 // the picker on a whole sample, on the user's desktop: a long task
  Object.assign(h.priors.settings.xenium, XENIUM);
  let r = h.draw();
  const { NEAREST, LINEAR, TEXTURE_MIN_FILTER, TEXTURE_MAG_FILTER, TEXTURE_WRAP_S, TEXTURE_WRAP_T, CLAMP_TO_EDGE } = h.stage.gl;
  const filters = log => named(log, 'texParameteri').filter(c => [TEXTURE_MIN_FILTER, TEXTURE_MAG_FILTER].includes(c[1][1])).map(c => c[1][2]);
  assert.deepEqual(filters(r.gl), [NEAREST, NEAREST], 'drawn at its camera: texel for pixel');
  assert.deepEqual(named(r.gl, 'texParameteri').filter(c => [TEXTURE_WRAP_S, TEXTURE_WRAP_T].includes(c[1][1])).map(c => c[1][2]),
    [CLAMP_TO_EDGE, CLAMP_TO_EDGE], 'clamped once, when made: a moved raster blends nothing in from its opposite edge');
  assert.equal(strokes(r.calls).length, 1); assert.equal(named(r.gl, 'texImage2D').length, 1);
  assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1, 1, 1, -1], 'drawn over the whole stage');
  r = h.draw();
  assert.deepEqual([strokes(r.calls).length, named(r.gl, 'texImage2D').length, named(r.gl, 'drawArrays').length], [0, 0, 1], 'unchanged: only the quad');
  // A pan of 10 µm to the right at zoom 2 on an 800 px stage, while the camera is moving (Stage.emit: movingUntil).
  h.stage.movingUntil = h.w.now + 140; h.stage.camera.x += 10;
  r = h.draw();
  assert.deepEqual([strokes(r.calls).length, named(r.gl, 'texImage2D').length], [0, 0], 'no stroke, no upload while moving');
  assert.deepEqual(filters(r.gl), [LINEAR, LINEAR], 'moved under the camera: sampled linearly (no blocks, no dropped rows)');
  const shift = 2 * 10 * 2 / 800;
  named(r.gl, 'uniform4fv')[0][1][1].forEach((v, k) => assert.ok(Math.abs(v - [-1 - shift, 1, 1 - shift, -1][k]) < 1e-12, 'the raster moved with the camera'));
  // Zoom x2 about the same centre and a narrower stage (the UMAP pane), still moving.
  h.stage.movingUntil = h.w.now + 140; h.stage.camera.x -= 10; h.stage.camera.zoom = 4;
  r = h.draw();
  named(r.gl, 'uniform4fv')[0][1][1].forEach((v, k) => assert.ok(Math.abs(v - [-2, 2, 2, -2][k]) < 1e-12));
  h.stage.movingUntil = h.w.now + 140; h.stage.canvas.width = 500;
  r = h.draw();
  assert.equal(strokes(r.calls).length, 0);
  const due = h.w.timers.filter(t => !t.cleared && !t.fired);
  assert.equal(due.length, 1, 'one settle timer, pushed back by every step');
  h.w.advance(100); assert.equal(h.stage.dirty, false, 'not yet');
  h.w.advance(50); assert.equal(h.stage.dirty, true, 'the stage is woken once it is still');
  r = h.draw();
  assert.deepEqual([strokes(r.calls).length, named(r.gl, 'texImage2D').length], [1, 1]);
  assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1, 1, 1, -1]);
  assert.deepEqual(filters(r.gl), [NEAREST, NEAREST]);
  assert.deepEqual(r.calls.find(c => c[0] === 'setTransform')[1], [4, 0, 0, 4, 250 - 500 * 4, 300 - 500 * 4], 'at the settled camera and size');
  // A fit tween moves the camera without a pointer: it counts as moving too.
  h.stage.tween = {}; h.stage.camera.y += 5;
  assert.equal(strokes(h.draw().calls).length, 0);
  h.stage.tween = null;
  assert.equal(strokes(h.draw().calls).length, 1, 'a moved camera that is no longer moving is drawn at once');
});

test('what the outlines show (settings, vectors, visibility) is drawn at once, even while the camera moves', () => {
  const h = setup();
  Object.assign(h.priors.settings.xenium, XENIUM);
  h.draw();
  h.stage.movingUntil = Infinity; h.stage.camera.x += 3;
  for (const change of [() => { h.priors.settings.xenium.opacity = .4; }, () => { h.priors.settings.xenium.color = '#ff0000'; },
    () => { h.priors.settings.xenium.width = 2.5; }, () => { h.priors.settings.nuclei.on = true; },
    () => { h.priors.vectors = { cells: grid(5, 5), nuclei: null }; }]) {
    h.w.with(change);
    const r = h.draw();
    assert.equal(named(r.gl, 'texImage2D').length, 1);
    assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1, 1, 1, -1]);
  }
  h.priors.settings.xenium.on = false; h.priors.settings.nuclei.on = false;
  assert.deepEqual(h.draw().gl, [], 'nothing on: no upload, no quad');
  h.priors.settings.xenium.on = true; h.priors.vectorsVisible = false;
  assert.deepEqual(h.draw().gl, [], 'a whole-sample overview hides the outlines');
  h.w.with(() => h.priors.reset());
  assert.ok(h.w.timers.every(t => t.cleared || t.fired), 'reset drops a pending settle');
});

test('each drawing is a plv:prior-rebuild user-timing measure with its polygons and layers, cleared from the timeline at once', () => {
  const h = setup({ uploadMs: 120 });
  Object.assign(h.priors.settings.xenium, XENIUM);
  h.w.now = 1000; h.draw();
  assert.equal(h.w.measures.length, 1);
  const m = h.w.measures[0];
  assert.equal(m.name, 'plv:prior-rebuild'); assert.equal(m.detail.layer, 'xenium'); assert.ok(m.detail.polygons > 100);
  assert.ok(m.start < m.end, 'the upload included');
  h.priors.settings.nuclei.on = true; h.draw();
  assert.equal(h.w.measures[1].detail.layer, 'xenium+nuclei');
  assert.deepEqual(h.w.cleared, ['plv:prior-rebuild', 'plv:prior-rebuild'], 'observers have them; the page keeps none');
  h.stage.movingUntil = Infinity; h.stage.camera.x += 1; h.draw(); h.draw();
  assert.equal(h.w.measures.length, 2, 'a raster moved with the camera is no drawing');
});

test('the picker (no run info) strokes the layers it was given, whatever the run\'s layer settings', () => {
  const h = setup({ info: null });
  h.priors.settings.xenium.on = false; h.priors.settings.nuclei.on = false;
  assert.equal(strokes(h.draw().calls).length, 2, 'both chosen shapes priors');
  h.w.with(() => { h.priors.vectors = { cells: grid(30, 30), nuclei: null }; });
  assert.equal(strokes(h.draw().calls).length, 1, 'a nuclei prior that is not shapes is not stroked');
  h.stage.info = { cells: 1 };
  assert.equal(strokes(h.draw().calls).length, 0, 'in a run the layer settings decide');
});

test('vectors: legacy JSON is converted once on assignment, metadata kept; PolygonSets are kept as they are', () => {
  const h = setup();
  const v = h.priors.vectors;
  assert.equal(v.cells.count, 900); assert.equal(v.nuclei.count, 129); assert.deepEqual(v.source, { cells: 'xenium_cell_boundaries' });
  const set = polygonSetFromLegacy(grid(3, 3)), binary = { cells: set, nuclei: null, source: {} };
  h.priors.vectors = binary;
  assert.equal(h.priors.vectors, binary, 'nothing to convert: the object itself');
  assert.equal(h.priors.vectors.cells, set);
  h.priors.vectors = null;
  assert.equal(h.priors.vectors, null);
  assert.equal(h.priors.visible(), false);
});

// ---- drawing while moving, the picker's hold, memory, and the paths not covered before --------------------------------
test('a cheap drawing is drawn again while the camera moves, within a tenth of the main thread', () => {
  const h = setup({ uploadMs: 4 });                   // paint and upload: 4 ms on the test clock
  Object.assign(h.priors.settings.xenium, XENIUM);
  h.draw();
  const { NEAREST, TEXTURE_MIN_FILTER, TEXTURE_MAG_FILTER } = h.stage.gl, drawn = [];
  // A drag: a camera step every 16 ms frame, for half a second.
  for (let frame = 0; frame < 30; frame++) {
    h.w.now += 16; h.stage.camera.x += 2; h.stage.movingUntil = h.w.now + 140;
    const r = h.draw();
    drawn.push(strokes(r.calls).length);
    if (strokes(r.calls).length) {
      assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1, 1, 1, -1], 'drawn at the camera of this frame');
      assert.deepEqual(named(r.gl, 'texParameteri').filter(c => [TEXTURE_MIN_FILTER, TEXTURE_MAG_FILTER].includes(c[1][1])).map(c => c[1][2]), [NEAREST, NEAREST]);
      assert.equal(r.calls.find(c => c[0] === 'setTransform')[1][4], 400 - h.stage.camera.x * 2);
    }
  }
  const count = drawn.reduce((a, b) => a + b, 0);
  assert.ok(count >= 9 && count <= 13, `a fresh drawing every ~40 ms (4 ms each): ${count} in 30 frames`);
  assert.ok(drawn.every((n, k) => !n || !drawn[k + 1]), 'never in two frames running');
  // It stops where the last drawing is already at the camera: nothing to settle.
  while (!strokes(h.draw().calls).length) { h.w.now += 16; h.stage.camera.x += 2; h.stage.movingUntil = h.w.now + 140; }
  h.w.advance(1000);
  assert.equal(strokes(h.draw().calls).length, 0, 'the view drawn last is the settled one');
});

test('the picker holds the old box\'s outlines while the next box loads: one drawing per pan, not two', () => {
  const h = setup({ info: null, uploadMs: 120 });     // the picker on a whole sample (no run): every drawing a long task
  h.draw();
  const paints = [];
  const step = () => { const r = h.draw(); paints.push(strokes(r.calls).length ? 'paint' : named(r.gl, 'drawArrays').length ? 'moved' : '-'); };
  // A right-button pan: main.js keeps the old box's outlines and holds them while it fetches the new box.
  h.priors.hold = true;
  for (let k = 0; k < 5; k++) { h.w.now += 16; h.stage.camera.x += 5; h.stage.movingUntil = h.w.now + 140; step(); }
  assert.equal(h.w.timers.filter(t => !t.cleared && !t.fired).length, 0, 'no settle timer while held');
  h.w.advance(500); step();                           // at rest, still waiting for the new box (the 200 ms debounce, then the fetch)
  assert.deepEqual(paints, ['moved', 'moved', 'moved', 'moved', 'moved', 'moved'], 'the old outlines are moved, never drawn again');
  h.w.with(() => { h.priors.vectors = { cells: grid(30, 30), nuclei: null }; h.priors.hold = false; });
  step();
  assert.deepEqual(paints.slice(6), ['paint'], 'the new box: drawn once');
  h.w.advance(1000); step();
  assert.equal(paints.at(-1), 'moved', 'and only once');
});

test('vectors replaced: the old sets\' tiles and grids go at once; a set keeps tiles within its budget, least recently used first', () => {
  const set = polygonSetFromLegacy(grid(40, 40, 10)), all = [-1e4, -1e4, 1e4, 1e4];
  const built = () => RecPath.made.length;
  RecPath.made.length = 0;
  layerPath(set, all, 1, RecPath);
  const first = built();
  layerPath(set, all, 1, RecPath);
  assert.equal(built() - first, 1, 'tiles kept: only the view\'s own path is new');
  releaseTiles(set);
  layerPath(set, all, 1, RecPath);
  assert.equal(built() - first - 1, first, 'released: every tile is built again');
  // A budget of 3,000 vertices (the grid's 1,600 squares hold 8,000): far views keep only what fits.
  const small = polygonSetFromLegacy(grid(40, 40, 10)), left = [0, 0, 120, 400], right = [280, 0, 400, 400];
  RecPath.made.length = 0;
  const a = layerPath(small, left, 1, RecPath, false, 3000), madeLeft = built();
  layerPath(small, right, 1, RecPath, false, 3000);
  const before = built();
  layerPath(small, left, 1, RecPath, false, 3000);
  assert.ok(built() - before > 1, 'the left tiles were evicted for the right ones: built again');
  assert.ok(a.polygons > 0 && madeLeft > 1);
  // PriorLayers lets go of a replaced set's tiles: a replaced set is built again if it is ever shown again.
  const h = setup();
  Object.assign(h.priors.settings.xenium, XENIUM);
  h.draw();
  const old = h.priors.vectors.cells;
  h.w.with(() => { h.priors.vectors = { cells: grid(5, 5), nuclei: null }; });
  RecPath.made.length = 0;
  layerPath(old, all, 2, RecPath);
  assert.ok(RecPath.made.length > 2, 'the replaced set\'s tiles were released');
});

test('while moving, the path of the same tiles is reused; a settled view always gets its own', () => {
  const set = polygonSetFromLegacy(grid(40, 40, 10)), box = [0, 0, 200, 200];
  const a = layerPath(set, box, 1, RecPath, true), b = layerPath(set, [1, 1, 201, 201], 1, RecPath, true);
  assert.equal(b.path, a.path, 'the same tiles: the same path object (no new native path per frame)');
  const c = layerPath(set, [1, 1, 201, 201], 1, RecPath, false);
  assert.notEqual(c.path, a.path, 'a settled view: a path of its own');
  assert.deepEqual(c.path.ops, a.path.ops);
  const d = layerPath(set, [0, 0, 400, 400], 1, RecPath, true);
  assert.notEqual(d.path, c.path, 'other tiles: another path');
});

test('a device pixel ratio change redraws at once, at the new line width; outlines shown again after a camera move are drawn at that camera', () => {
  const h = setup();
  Object.assign(h.priors.settings.xenium, XENIUM);
  h.draw();
  h.stage.ratio = 2;
  let r = h.draw();
  assert.equal(strokes(r.calls).length, 1);
  assert.equal(strokes(r.calls)[0][2].lineWidth, 1 * 2 / 2, 'width x ratio / zoom');
  h.priors.vectorsVisible = false; h.draw();
  h.stage.camera.x += 50; h.stage.camera.zoom = 3;
  h.priors.vectorsVisible = true;
  r = h.draw();
  assert.equal(strokes(r.calls).length, 1);
  assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1, 1, 1, -1]);
  assert.deepEqual(r.calls.find(c => c[0] === 'setTransform')[1], [3, 0, 0, 3, 400 - 550 * 3, 300 - 500 * 3]);
});

test('a context restored while the camera moves uploads the raster on screen, where it belongs', () => {
  const h = setup({ uploadMs: 120 });
  Object.assign(h.priors.settings.xenium, XENIUM);
  h.draw();
  h.stage.movingUntil = h.w.now + 1000; h.stage.camera.x += 10;
  h.draw();
  const log = [];
  h.stage.gl = fakeGL(log, h.charge); h.stage.imageProgram = {};       // webglcontextrestored: a new context
  h.w.calls.length = 0;
  h.w.with(() => h.priors.draw());
  assert.equal(named(log, 'texImage2D').length, 1, 'the shown raster is uploaded to the new context');
  assert.equal(named(log, 'createProgram').length, 1);
  assert.equal(h.w.calls.filter(c => c[0] === 'stroke').length, 0, 'not drawn again while moving');
  const clip = named(log, 'uniform4fv').at(-1)[1][1], shift = 2 * 10 * 2 / 800;
  clip.forEach((v, k) => assert.ok(Math.abs(v - [-1 - shift, 1, 1 - shift, -1][k]) < 1e-12, 'with the moved camera\'s clip'));
});

test('a run of camera steps keeps one settle timer, and the view settles once', () => {
  const h = setup({ uploadMs: 120 });
  Object.assign(h.priors.settings.xenium, XENIUM);
  h.draw();
  for (let k = 0; k < 10; k++) { h.w.now += 30; h.stage.camera.x += 1; h.stage.movingUntil = h.w.now + 140; h.draw(); }
  assert.equal(h.w.timers.filter(t => !t.cleared && !t.fired).length, 1);
  h.w.advance(200);
  assert.equal(strokes(h.draw().calls).length, 1);
  h.w.advance(1000);
  assert.equal(strokes(h.draw().calls).length, 0, 'settled once');
});

test('a detail run loads its prior vectors once, as the binary, and its prior raster once its info has it', async t => {
  const h = setup(), asked = [], binary = encodeVectors({ cells: grid(3, 2), nuclei: null }, { source: { cells: 'cells' } });
  let failVectors = true;
  t.mock.method(globalThis, 'fetch', async url => {
    asked.push(url);
    if (url.endsWith(`/xenium?${VECTORS_QUERY}`)) return failVectors ? new Response('busy', { status: 503 }) : new Response(binary.slice(0));
    if (url.endsWith('/prior')) return new Response(new ArrayBuffer(8));
    return new Response('{}', { status: 404 });
  });
  h.priors.rebuild = () => { h.priors.rebuilt = (h.priors.rebuilt || 0) + 1; };
  h.w.with(() => h.priors.reset());
  const run = { id: 'r1', mode: 'detail', info: null };
  await h.priors.load(run);                       // openRun: no info yet (still preparing); the engine is busy
  assert.deepEqual(asked, [`/api/v1/runs/r1/xenium?${VECTORS_QUERY}`], 'the binary, never the JSON');
  failVectors = false;
  run.info = { prior: { nx: 1, ny: 1 } };
  await h.priors.load(run);                       // onInfo: the info has come; the failed vectors are asked again
  await h.priors.load(run);                       // openRun's second call for a run that had its info already
  assert.equal(asked.filter(u => u.includes('/xenium')).length, 2);
  assert.equal(asked.filter(u => u.endsWith('/prior')).length, 1, 'the prior raster once');
  assert.equal(h.priors.vectors.cells.count, 6); assert.deepEqual(h.priors.vectors.source, { cells: 'cells' });
  assert.equal(h.priors.rebuilt, 1);
  h.w.with(() => h.priors.reset());              // another run (or the same opened again)
  await h.priors.load({ id: 'r2', mode: 'detail', info: null });
  assert.equal(asked.at(-1), `/api/v1/runs/r2/xenium?${VECTORS_QUERY}`);
  const before = asked.length;
  await h.priors.load({ id: 't', mode: 'tiled', info: { prior: {} } });
  assert.equal(asked.length, before, 'whole-sample runs: TiledLoader loads them');
});

const bitmap = () => ({ closed: 0, close() { this.closed++; } });

test('the picker uploads one complete worker bitmap, without main-thread strokes or repeated uploads', () => {
  const h = setup({ info: null, worker: true, uploadMs: 7 });
  let r = h.draw();
  const worker = h.w.workers[0], first = bitmap();
  assert.ok(worker.url.href.endsWith('/prior-raster-worker.mjs'));
  assert.deepEqual(worker.options, { type: 'module' });
  assert.deepEqual(r.calls, [], 'no main-thread canvas operations');
  assert.equal(named(r.gl, 'texImage2D').length, 0, 'nothing incomplete is uploaded');
  assert.equal(worker.requests[0].vectors.cells.count, 900);
  assert.equal(h.priors.vectors.cells.xy.byteLength > 0, true, 'source arrays are still owned by the app');
  h.draw(); h.draw();
  assert.equal(worker.requests.length, 1, 'one in-flight job');
  h.w.advance(250); worker.reply(first);
  assert.equal(h.stage.dirty, true, 'completion wakes a sleeping stage');
  r = h.draw();
  assert.equal(named(r.gl, 'texImage2D')[0][1].at(-1), first);
  assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1, 1, 1, -1]);
  assert.equal(h.priors.cost, 247, 'worker cost plus upload, without the scheduling delay');
  assert.equal(h.w.measures[0].end - h.w.measures[0].start, 257, 'the user timing is end-to-end latency');
  assert.deepEqual(h.w.cleared, ['plv:prior-rebuild']);
  assert.equal(first.closed, 0, 'kept for WebGL context restoration');
  r = h.draw();
  assert.deepEqual([r.calls.length, named(r.gl, 'texImage2D').length, named(r.gl, 'drawArrays').length], [0, 0, 1]);
  assert.equal(worker.requests.length, 1);
});

test('a worker picker pan holds and moves the previous texture until a complete replacement is ready', () => {
  const h = setup({ info: null, worker: true });
  h.draw(); const worker = h.w.workers[0], first = bitmap(); worker.reply(first); h.draw();
  h.priors.hold = true;
  h.stage.camera.x += 10; h.stage.movingUntil = h.w.now + 140;
  let r = h.draw();
  assert.equal(worker.requests.length, 1, 'no old-box redraw while fetching');
  assert.equal(named(r.gl, 'texImage2D').length, 0);
  assert.equal(named(r.gl, 'drawArrays').length, 1);
  assert.equal(named(r.gl, 'texParameteri').at(-1)[1][2], h.stage.gl.LINEAR);
  const shift = 2 * 10 * 2 / 800;
  assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1 - shift, 1, 1 - shift, -1]);
  h.w.advance(200); h.draw();
  assert.equal(worker.requests.length, 1);
  h.w.with(() => { h.priors.vectors = { cells: grid(10, 10), nuclei: null }; h.priors.hold = false; });
  r = h.draw();
  assert.equal(worker.requests.length, 2);
  assert.equal(worker.requests[1].vectors.cells.count, 100);
  assert.equal(first.closed, 0);
  assert.equal(named(r.gl, 'texImage2D').length, 0, 'the old complete texture stays until its replacement arrives');
  const second = bitmap(); worker.reply(second); r = h.draw();
  assert.equal(named(r.gl, 'texImage2D')[0][1].at(-1), second);
  assert.equal(first.closed, 1);
  assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1, 1, 1, -1]);
  h.stage.camera.y += 10; h.draw();
  assert.equal(worker.requests.length, 3);
  assert.equal('vectors' in worker.requests[2], false, 'the same vectors are not cloned for each view');
});

test('worker requests coalesce changes and discard obsolete camera, size, DPR, style and vector results', () => {
  for (const change of [
    h => { h.stage.camera.x++; }, h => { h.stage.camera.zoom *= 2; },
    h => { h.stage.canvas.width++; }, h => { h.stage.canvas.height++; }, h => { h.stage.ratio = 2; },
    h => { h.priors.settings.nuclei.width = 3; },
    h => { h.priors.vectors = { cells: grid(2, 2), nuclei: null }; },
  ]) {
    const h = setup({ info: null, worker: true });
    h.draw(); const worker = h.w.workers[0];
    h.w.with(() => change(h)); h.draw(); h.draw();
    assert.equal(worker.requests.length, 1, 'do not queue or clone behind the running job');
    const stale = bitmap(); worker.reply(stale);
    let r = h.draw();
    assert.equal(stale.closed, 1);
    assert.equal(named(r.gl, 'texImage2D').length, 0, 'no stale image is installed');
    assert.equal(worker.requests.length, 2, 'request the latest state after completion');
    const current = bitmap(); worker.reply(current); r = h.draw();
    assert.equal(named(r.gl, 'texImage2D')[0][1].at(-1), current);
    assert.equal(h.priors.shown.view.zoom, h.stage.camera.zoom);
    assert.equal(h.priors.shown.view.ratio, h.stage.ratio);
    assert.equal(h.priors.shown.vectors, h.priors.vectors);
  }
});

test('returning to the shown view during a worker rebuild keeps its unchanged texture without another upload', () => {
  const h = setup({ info: null, worker: true });
  h.draw(); const worker = h.w.workers[0], first = bitmap(); worker.reply(first); h.draw();
  h.stage.camera.x += 10; h.draw();
  h.stage.camera.x -= 10; h.draw();
  const stale = bitmap(); worker.reply(stale); const r = h.draw();
  assert.equal(stale.closed, 1);
  assert.equal(first.closed, 0);
  assert.equal(h.priors.bitmap, first);
  assert.equal(worker.requests.length, 2);
  assert.equal(named(r.gl, 'texImage2D').length, 0);
  assert.equal(named(r.gl, 'drawArrays').length, 1);
});

test('a late worker image cannot revive a reset, hidden, orbit or replaced picker', () => {
  for (const change of [
    h => h.priors.reset(), h => { h.priors.vectorsVisible = false; },
    h => { h.priors.vectors = null; }, h => { h.stage.orbit.active = true; },
    h => { h.stage.info = { cells: 1 }; h.priors.settings.xenium.on = true; },
  ]) {
    const h = setup({ info: null, worker: true });
    h.draw(); const worker = h.w.workers[0], completed = bitmap(); worker.reply(completed); h.draw();
    h.stage.camera.x++; h.draw();
    h.w.with(() => change(h)); h.draw();
    assert.equal(worker.terminated, true);
    assert.equal(completed.closed, 1);
    h.stage.dirty = false;
    const late = bitmap(); worker.reply(late);
    assert.equal(late.closed, 1);
    assert.equal(h.stage.dirty, false, 'an obsolete worker does not wake the stage');
    assert.equal(h.priors.ready, null);
  }
});

test('reset closes an unconsumed result and ignores results from a previous worker even after another starts', () => {
  const h = setup({ info: null, worker: true });
  h.draw(); const old = h.w.workers[0], waiting = bitmap(); old.reply(waiting);
  h.w.with(() => h.priors.reset());
  assert.equal(waiting.closed, 1);
  h.w.with(() => { h.priors.vectors = { cells: grid(2, 2), nuclei: null }; });
  h.draw(); const current = h.w.workers[1], late = bitmap();
  old.reply(late, { id: current.requests[0].id });
  assert.equal(late.closed, 1, 'worker identity is checked as well as job id');
  assert.ok(h.priors.pending);
  const result = bitmap(); current.reply(result); h.draw();
  assert.equal(h.priors.bitmap, result);
});

test('a restored WebGL context reuploads the retained worker bitmap at its camera without repainting it', () => {
  const h = setup({ info: null, worker: true });
  h.draw(); const worker = h.w.workers[0], first = bitmap(); worker.reply(first); h.draw();
  h.stage.camera.x += 10; h.stage.movingUntil = h.w.now + 140;
  h.stage.gl = fakeGL(h.log); h.stage.imageProgram = {};
  const r = h.draw();
  assert.equal(named(r.gl, 'texImage2D')[0][1].at(-1), first);
  assert.deepEqual(named(r.gl, 'uniform4fv')[0][1][1], [-1.05, 1, .95, -1]);
  assert.equal(worker.requests.length, 1);
  assert.equal(first.closed, 0);
  assert.deepEqual(r.calls, []);
});

test('worker errors fall back once to the existing canvas renderer and release worker resources', () => {
  for (const fail of [
    worker => worker.reply(undefined, { error: 'no 2-D context' }),
    worker => worker.onerror({ preventDefault() {} }),
    worker => worker.onmessageerror(),
  ]) {
    const h = setup({ info: null, worker: true });
    h.draw(); const worker = h.w.workers[0]; h.w.with(() => fail(worker));
    assert.equal(worker.terminated, true);
    let r = h.draw();
    assert.equal(strokes(r.calls).length, 2);
    assert.equal(named(r.gl, 'texImage2D')[0][1].at(-1), h.priors.canvas);
    r = h.draw();
    assert.equal(h.w.workers.length, 1, 'do not restart an unsupported or broken worker on every draw');
    assert.equal(named(r.gl, 'texImage2D').length, 0);
  }
});

test('worker construction and posting failures use the synchronous renderer in the same frame', () => {
  const h = setup({ info: null, worker: true });
  h.w.with(() => {
    const saved = globalThis.Worker;
    globalThis.Worker = class { constructor() { throw Error('CSP'); } };
    h.priors.draw(); globalThis.Worker = saved;
  });
  assert.equal(strokes(h.w.calls).length, 2);
  assert.equal(h.priors.workerFailed, true);

  const p = setup({ info: null, worker: true });
  p.draw(); const worker = p.w.workers[0], first = bitmap(); worker.reply(first); p.draw();
  worker.postMessage = () => { throw Error('clone failed'); };
  p.stage.camera.x++; const r = p.draw();
  assert.equal(first.closed, 1);
  assert.equal(worker.terminated, true);
  assert.equal(strokes(r.calls).length, 2);
  assert.equal(named(r.gl, 'texImage2D')[0][1].at(-1), p.priors.canvas);
});

test('a browser without OffscreenCanvas keeps the synchronous picker path', () => {
  const h = setup({ info: null, worker: true });
  h.w.with(() => { delete globalThis.OffscreenCanvas; h.priors.draw(); });
  assert.equal(strokes(h.w.calls).length, 2);
  assert.equal(named(h.log, 'texImage2D')[0][1].at(-1), h.priors.canvas);
  assert.equal(h.w.workers.length, 0);
});

test('detail views and a visible prior raster keep the original synchronous renderer', () => {
  const detail = setup({ worker: true });
  Object.assign(detail.priors.settings.xenium, XENIUM);
  assert.equal(strokes(detail.draw().calls).length, 1);
  assert.equal(detail.w.workers.length, 0);
  const prior = setup({ worker: true, info: null });
  prior.priors.prior = recCanvas([]); prior.priors.meta = { x0: 10, y0: 20, nx: 4, ny: 3, voxel: 2 };
  const r = prior.draw();
  assert.equal(prior.w.workers.length, 0);
  assert.equal(strokes(r.calls).length, 2);
  assert.ok(r.calls.some(c => c[0] === 'drawImage'));
});

/** prior-layers.js with its harness hook bound to `hook` (as harness-hook.test.mjs loads modules: every other import is
 *  the real module). */
async function hookedLayers(hook) {
  const SRC = new URL('../src/', import.meta.url), dataURL = text => `data:text/javascript;base64,${Buffer.from(text).toString('base64')}`;
  globalThis.__priorLayersHook = hook;
  const load = ++hookedLayers.loads, file = new URL('prior-layers.js', SRC);   // a fresh stand-in per hook (modules are cached by URL)
  const source = (await readFile(file, 'utf8')).replace(/from '\.\/([^']+)'/g, (_, name) =>
    `from ${JSON.stringify(name === 'harness-hook.mjs' ? dataURL(`export const harness = globalThis.__priorLayersHook; // ${load}`) : new URL(name, SRC).href)}`)
    .replaceAll('import.meta.url', JSON.stringify(file.href));   // the worker's URL resolves against the real file
  return (await import(dataURL(`${source}\n// ${load}`))).PriorLayers;
}

test('the hook gets the main thread\'s part of a worker rebuild (posting plus uploading) and the worker\'s drawing apart', async () => {
  const calls = [], hook = { priorRebuild: (ms, n) => calls.push(['main', ms, n]), priorWorkerPaint: (ms, n) => calls.push(['worker', ms, n]) };
  const h = setup({ info: null, worker: true, uploadMs: 7, postMs: 3, Layers: await hookedLayers(hook) });
  h.draw(); const worker = h.w.workers[0];
  assert.deepEqual(calls, [], 'nothing before the image exists');
  h.w.advance(250); worker.reply(bitmap(), { paintMs: 56, polygons: 6456 });
  assert.deepEqual(calls, [['worker', 56, 6456]], 'the worker\'s drawing, on arrival: off the main thread');
  h.draw();
  assert.deepEqual(calls, [['worker', 56, 6456], ['main', 3 + 7, 6456]], 'posting the request (3 ms) and the upload (7 ms)');
  h.draw();
  assert.equal(calls.length, 2, 'an unchanged view reports nothing');
  // An image for a camera that has moved on is drawn by the worker but never uploaded: no main-thread rebuild.
  h.stage.camera.x += 10; h.draw(); h.stage.camera.x += 10; h.draw();
  worker.reply(bitmap(), { paintMs: 40, polygons: 900 }); h.draw();
  assert.deepEqual(calls.slice(2), [['worker', 40, 900]]);
  worker.reply(bitmap(), { paintMs: 41, polygons: 901 }); h.draw();
  assert.deepEqual(calls.slice(3), [['worker', 41, 901], ['main', 3 + 7, 901]]);
});

test('the synchronous renderer reports path building and stroking as before, never a worker paint; a hook without the worker counter still works', async () => {
  const calls = [], hook = { priorRebuild: (ms, n) => calls.push(['main', ms, n]), priorWorkerPaint: (ms, n) => calls.push(['worker', ms, n]) };
  const detail = setup({ worker: true, uploadMs: 7, Layers: await hookedLayers(hook) });
  Object.assign(detail.priors.settings.xenium, XENIUM);
  detail.draw();
  assert.equal(detail.w.workers.length, 0);
  assert.deepEqual(calls.map(c => c[0]), ['main'], 'one rebuild, on the main thread');
  assert.ok(calls[0][2] > 100 && calls[0][2] <= 900, 'the polygons stroked (those near the view)');
  assert.equal(calls[0][1], 0, 'stroking costs nothing on the test clock: the 7 ms upload is not in it');
  const older = [], h = setup({ info: null, worker: true, Layers: await hookedLayers({ priorRebuild: (ms, n) => older.push([ms, n]) }) });
  h.draw(); h.w.workers[0].reply(bitmap());
  assert.doesNotThrow(() => h.draw());
  assert.deepEqual(older, [[0, 1000]], 'only the main-thread part, through the counter it has');
});
hookedLayers.loads = 0;

test('stain tile and resolved vectors are painted by the worker and uploaded after a pan', () => {
  const h = setup({info:{kind:'stain'},worker:true});
  Object.assign(h.priors.settings.xenium,XENIUM);
  h.priors.settings.nuclei.on = false;
  const cells = [{id:1,poly:[[490,490],[510,490],[510,510],[490,510],[490,490]]}];
  for (const revision of [1,2]) {
    const vectors = decodeVectors(encodeVectors({cells},{revision,resolved:revision===2}));
    h.w.with(()=>{h.priors.vectors=vectors;});
    h.draw();
    const worker = h.w.workers[0];
    assert.ok(worker,'stain runs use the outline worker even with stage.info set');
    assert.equal(worker.requests.at(-1).vectors.cells.count,1);
    assert.notEqual(h.priors.shown?.vectors,vectors,'fetching is not painting');
    worker.reply(bitmap(),{polygons:1});
    const painted = h.draw();
    assert.equal(h.priors.shown.vectors,vectors);
    assert.equal(named(painted.gl,'texImage2D').length,1);
    assert.equal(named(painted.gl,'drawArrays').length,1);
    h.stage.camera.x += 5;
    h.draw(); worker.reply(bitmap(),{polygons:1}); h.draw();
    assert.equal(h.priors.shown.vectors,vectors);
    assert.equal(h.priors.shown.view.x,h.stage.camera.x);
  }
});

// ---- Go deeper's starting outline of a cell ("the prior layer's polygon with the
// cell's name, or, without one, the outline of the cell's labels at the start"; Stage.setStartOutline draws it) ----------
/** A label raster from rows of labels (row j = y voxel j). */
const raster = (rows, x0 = 0, y0 = 0, voxel = 1) => ({ labels: Uint32Array.from(rows.flat()), nx: rows[0].length, ny: rows.length, x0, y0, voxel });
/** The ring's corners as [x, y] pairs, and its area (shoelace, in the raster's µm²). */
const pairs = ring => Array.from({ length: ring.length / 2 }, (_, k) => [ring[2 * k], ring[2 * k + 1]]);
const area = ring => Math.abs(pairs(ring).reduce((a, [x, y], k, p) => a + x * p[(k + 1) % p.length][1] - p[(k + 1) % p.length][0] * y, 0)) / 2;

test('labelOutline: the outer boundary of a label\'s largest part along the voxel edges, corners only, in µm', () => {
  // One voxel of 2 µm at (1, 1) of a raster at (10, 20) µm: its four corners, clockwise on screen (y down).
  assert.deepEqual(pairs(labelOutline(raster([[0, 0, 0], [0, 9, 0], [0, 0, 0]], 10, 20, 2), 9)), [[14, 22], [14, 24], [12, 24], [12, 22]]);
  // An L: six corners (none in the middle of a straight run), its area the three voxels'.
  const L = labelOutline(raster([[0, 0, 0, 0], [0, 5, 0, 0], [0, 5, 5, 0], [0, 0, 0, 0]]), 5);
  assert.deepEqual(pairs(L), [[2, 1], [2, 2], [3, 2], [3, 3], [1, 3], [1, 1]]); assert.equal(area(L), 3);
  // Two parts: the larger one (a 2 x 2 block, not the voxel apart; the mask draws them as two outlines too).
  const two = labelOutline(raster([[4, 0, 0, 0], [0, 0, 4, 4], [0, 0, 4, 4]]), 4);
  assert.deepEqual(pairs(two), [[4, 1], [4, 3], [2, 3], [2, 1]]);
  // Voxels touching at a corner only are two parts: a diagonal pair gives one voxel.
  assert.equal(area(labelOutline(raster([[3, 0], [0, 3]]), 3)), 1);
  // A hole: the outer boundary (a ring around the hole is not the cell's outline).
  const ring = labelOutline(raster([[6, 6, 6], [6, 0, 6], [6, 6, 6]]), 6);
  assert.deepEqual(pairs(ring), [[3, 0], [3, 3], [0, 3], [0, 0]]);
  // A hole that meets the outside at one corner: one ring around the cell and into the hole, its area the cell's own.
  const pinch = labelOutline(raster([[6, 6, 6], [6, 0, 6], [6, 6, 0]]), 6);
  assert.equal(area(pinch), 7); assert.equal(pinch.length % 2, 0);
  // At the raster's edges; none of it, label 0 or not a raster: null.
  assert.equal(area(labelOutline(raster([[7, 7], [7, 7]]), 7)), 4);
  for (const [r, label] of [[raster([[1, 2], [3, 4]]), 5], [raster([[0, 0], [0, 0]]), 0], [null, 1], [{ labels: new Uint32Array(3), nx: 2, ny: 2 }, 1]]) assert.equal(labelOutline(r, label), null);
});

/** PriorLayers with no stage drawing (startOutline reads only its vectors and its prior). */
function bare() {
  const w = world(), stage = { canvas: { width: 100, height: 100 }, camera: { x: 0, y: 0, zoom: 1 }, ratio: 1, info: { cells: 1 }, orbit: { active: false }, frame: null, dirty: false };
  return { w, priors: w.with(() => new PriorLayers(stage, () => 'rgb(1,2,3)')) };
}
const box = (x, y, s) => [[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]];

test('startOutline: the prior\'s polygon of the cell\'s name (its largest part), else its starting labels traced, else null; kept until they change', () => {
  const { w, priors } = bare();
  assert.equal(priors.startOutline('cellpose-2', 3), null, 'nothing loaded');
  // The binary's string names (a Cellpose start's cells: "cellpose-<id>", engine/zarr_bridge): the name's largest part.
  const cells = [{ id: 'cellpose-1', poly: box(0, 0, 10) }, { id: 'cellpose-2', poly: box(20, 0, 4) }, { id: 'cellpose-2', poly: box(30, 0, 12) }];
  w.with(() => { priors.vectors = decodeVectors(encodeVectors({ cells, nuclei: null })); });
  const got = priors.startOutline('cellpose-2', 3);
  assert.equal(got.source, 'prior'); assert.equal(got.closed, true);
  assert.deepEqual(Array.from(got.points), box(30, 0, 12).flat());
  assert.ok(got.points instanceof Float32Array && got.points.buffer !== priors.vectors.cells.xy.buffer, 'a copy, not a view of the set');
  assert.equal(priors.startOutline('cellpose-2', 3), got, 'asked again: the same answer, no work');
  // Numeric ids (a column prior's) match the name's digits.
  w.with(() => { priors.vectors = decodeVectors(encodeVectors({ cells: [{ id: 17, poly: box(5, 5, 3) }, { id: 170, poly: box(9, 9, 3) }], nuclei: null })); });
  assert.deepEqual(Array.from(priors.startOutline('17', 1).points), box(5, 5, 3).flat());
  w.with(() => { priors.vectors = { cells: polygonSetFromLegacy([{ id: 8, poly: box(1, 1, 2) }]), nuclei: null }; });
  assert.deepEqual(Array.from(priors.startOutline('8', 1).points), box(1, 1, 2).flat(), 'a legacy set\'s numeric ids too');
  // No polygon of that name: the prior's labels (a detail run's prior: label then permille per voxel, every layer; each
  // column's strongest), traced.
  const meta = { nx: 3, ny: 3, nz: 2, x0: 100, y0: 200, voxel: 2 }, n = 9;
  const prior = new Uint32Array(4 * n);
  prior.set([0, 0, 0, 0, 4, 4, 0, 0, 0], 0); prior.set([0, 0, 0, 0, 0, 0, 0, 4, 0], n);              // labels, layers 0 and 1
  prior.set([0, 0, 0, 0, 800, 800, 0, 0, 0], 2 * n); prior.set([0, 0, 0, 0, 0, 0, 0, 500, 0], 3 * n);  // permille
  Object.assign(priors, { meta, buffer: prior.buffer });
  const traced = priors.startOutline('cellpose-404', 4);
  assert.equal(traced.source, 'start');
  assert.deepEqual(pairs(traced.points), [[106, 202], [106, 204], [104, 204], [104, 206], [102, 206], [102, 202]], 'an L of three voxels of 2 µm');
  assert.equal(priors.startOutline('cellpose-404', 4), traced);
  assert.equal(priors.startOutline('cellpose-404', 6), null, 'a label the prior does not hold');
  // A whole-sample run's view of its prior (TiledLoader: setRaster's labels, the view's own geometry).
  Object.assign(priors, { buffer: null, raster: { labels: Uint32Array.of(9, 9, 0, 9), strength: new Uint32Array(4) }, meta: { nx: 2, ny: 2, x0: 0, y0: 0, voxel: 4 } });
  const view = priors.startOutline('x', 9);
  assert.equal(view.source, 'start'); assert.deepEqual(pairs(view.points), [[8, 0], [8, 8], [4, 8], [4, 4], [0, 4], [0, 0]]);
  // A run again (reset): nothing.
  w.with(() => priors.reset());
  assert.equal(priors.startOutline('x', 9), null);
});

// ---- Go deeper's hold split (its line's tooltip): how the start held each point's spot, from the run's own prior ----------
/** Points as a where reply holds them ({x, y, z} typed arrays). */
const pts = (xs, ys, zs = xs.map(() => NaN)) => ({ x: Float32Array.from(xs), y: Float32Array.from(ys), z: Float32Array.from(zs), count: xs.length });
/** A detail run's prior (label then permille per voxel, layer-major), from per-layer rows of [label, permille]. */
function heldPrior(layers, meta) {
  const n = meta.nx * meta.ny * meta.nz, out = new Uint32Array(2 * n);
  layers.flat(2).forEach(([label, permille], v) => { out[v] = label; out[n + v] = permille; });
  return out.buffer;
}

test('startHold: the nucleus or cell level of each point\'s spot at the start, by the worker\'s voxel rule; unknown on another label; levels from nucleus_prob and cell_prob, each ± 2', () => {
  const { w, priors } = bare();
  assert.equal(priors.startHold(pts([1], [1]), 4), null, 'nothing loaded');
  // 3 x 3 spots of 2 µm at (100, 200), two layers; proseg's levels: round(1000 x 0.8) = 800, round(1000 x 0.5) = 500.
  const meta = { nx: 3, ny: 3, nz: 2, x0: 100, y0: 200, voxel: 2, nucleus_prob: 0.8, cell_prob: 0.5 };
  const buffer = heldPrior([
    [[[0, 0], [4, 800], [4, 502]], [[4, 798], [4, 497], [7, 800]], [[0, 0], [0, 0], [0, 0]]],     // layer 0 (rows j = 0, 1, 2)
    [[[0, 0], [0, 0], [4, 650]], [[0, 0], [4, 800], [4, 500]], [[4, 800], [4, 500], [0, 0]]],     // layer 1
  ], meta);
  Object.assign(priors, { meta, buffer });
  const z = { grid_origin: 0, grid_spacing: 0.5 };   // info.z: layer floor(z / 0.5), clipped to the two layers
  const got = priors.startHold(pts(
    [102.5, 104.9, 100.1, 103, 105, 105, 103, 103, 99.9, 106, 103, 103],
    [200.5, 201, 202.1, 203, 203, 201, 203, 205, 201, 201, 205, 205],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.7, 0.7, 5, 0.2, 0.2, NaN, -1]), 4, z);
  assert.ok(got instanceof Uint8Array);
  assert.deepEqual([...got], [
    1,   // 800: the nucleus level
    2,   // 502: the cell level, within 2
    1,   // 798: the nucleus level, within 2
    0,   // 497: 3 from the cell level, neither
    0,   // label 7 there, 4 asked (another cell's spot, or a place rounded across an edge into it): not known
    0,   // 650 in layer 1: neither level
    1,   // layer 1 (z 0.7)
    2,   // z 5: clipped to the last layer
    0, 0,   // outside the grid (x 99.9: column -1; x 106: column 3)
    0,   // no depth in a two-layer prior: which layer can't be said
    0,   // z -1: clipped to layer 0, where no starting cell lay
  ]);
  // Another cell: its own spots only.
  assert.deepEqual([...priors.startHold(pts([105], [203], [0.2]), 7, z)], [1]);
  // Without info.z the depth grid is proseg's default (origin 0, spacing 1): z 0.7 lies in layer 0.
  assert.deepEqual([...priors.startHold(pts([103], [203], [0.7]), 4)], [0], 'layer 0 there: 497, neither');
  // A one-layer prior places a point without a depth.
  const flat = { ...meta, nz: 1 };
  Object.assign(priors, { meta: flat, buffer: heldPrior([[[[4, 800], [4, 500], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]], flat) });
  assert.deepEqual([...priors.startHold(pts([101, 103], [201, 201]), 4)], [1, 2]);
  // Levels proseg did not write (an older run's info): only the one it did can be told; equal levels: neither.
  priors.meta = { ...flat, nucleus_prob: undefined };
  assert.deepEqual([...priors.startHold(pts([101, 103], [201, 201]), 4)], [0, 2]);
  priors.meta = { ...flat, nucleus_prob: 0.5 };
  assert.deepEqual([...priors.startHold(pts([101, 103], [201, 201]), 4)], [0, 0], 'equal levels can\'t be told apart');
  priors.meta = { ...flat, nucleus_prob: 0.799, cell_prob: 0.801 };
  assert.deepEqual([...priors.startHold(pts([101], [201]), 4)], [0], 'within 2 of both: neither');
  // A whole-sample run (its view of the prior: setRaster, no buffer), a truncated prior, no label, no points: null.
  priors.meta = flat;
  assert.equal(priors.startHold(pts([101], [201]), 0), null); assert.equal(priors.startHold(null, 4), null);
  Object.assign(priors, { buffer: new ArrayBuffer(8) });
  assert.equal(priors.startHold(pts([101], [201]), 4), null);
  Object.assign(priors, { buffer: null, raster: { labels: Uint32Array.of(4, 4, 0, 0, 0, 0, 0, 0, 0), strength: new Uint32Array(9) } });
  assert.equal(priors.startHold(pts([101], [201]), 4), null, 'a whole-sample run: none');
  w.with(() => priors.reset());
  assert.equal(priors.startHold(pts([101], [201]), 4), null, 'another run: none');
});

test('startHold computes as the worker does, in 32-bit floats: a spot\'s column and layer at their edges are the engine\'s', () => {
  const { priors } = bare();
  // x0 0.1, 2 µm spots: x 2.1 is column 0 in 32-bit floats (the worker's float32 records), 1 in 64-bit ones.
  const meta = { nx: 3, ny: 1, nz: 5, x0: 0.1, y0: 0, voxel: 2, nucleus_prob: 0.8, cell_prob: 0.5 }, n = 15;
  const prior = new Uint32Array(2 * n);
  for (let v = 0; v < n; v++) { prior[v] = 9; prior[n + v] = v % 3 === 0 ? 800 : 500; }   // column 0: nucleus level
  Object.assign(priors, { meta, buffer: prior.buffer });
  assert.equal(Math.floor((2.1 - 0.1) / 2), 1, 'in 64-bit floats');
  assert.deepEqual([...priors.startHold(pts([2.1], [0.5], [0]), 9, { grid_origin: 0, grid_spacing: 0.1 })], [1], 'column 0: the nucleus level');
  // z 0.3 with layers of 0.1 from 0: layer 3 in 32-bit floats (2 in 64-bit ones); layer 3 of column 1 holds 500.
  const deep = new Uint32Array(2 * n);
  for (let v = 0; v < n; v++) { deep[v] = 9; deep[n + v] = Math.floor(v / 3) === 3 ? 800 : 500; }   // layer 3: nucleus level
  priors.buffer = deep.buffer;
  assert.equal(Math.floor(0.3 / 0.1), 2, 'in 64-bit floats');
  assert.deepEqual([...priors.startHold(pts([3], [0.5], [0.3]), 9, { grid_origin: 0, grid_spacing: 0.1 })], [1], 'layer 3: the nucleus level');
});

test('startHold reads a where reply\'s rounded place: one rounded across a spot\'s edge is read in the next spot, not known there unless that spot is the same cell\'s, whose level it then takes', () => {
  const { priors } = bare();
  // One layer of 3 x 1 spots of 2 µm from x 100: the cell (label 4) at the nucleus level in column 0 and at the cell
  // level in column 1, no cell in column 2. Two transcripts lie at x 101.996 and 103.996; a where reply gives places to
  // 0.01 µm, so they come as 102 and 104, each across its spot's edge.
  const meta = { nx: 3, ny: 1, nz: 1, x0: 100, y0: 200, voxel: 2, nucleus_prob: 0.8, cell_prob: 0.5 };
  Object.assign(priors, { meta, buffer: heldPrior([[[[4, 800], [4, 500], [0, 0]]]], meta) });
  assert.deepEqual([...priors.startHold(pts([101.996, 103.996], [201, 201]), 4)], [1, 2], 'where they lie: the nucleus level, the cell level');
  assert.deepEqual([...priors.startHold(pts([102, 104], [201, 201]), 4)], [2, 0],
    'as the reply gives them: the same cell\'s next spot, so its level (the cell level, not its own); then no cell\'s spot: not known');
});

// ---- the sample's segmentation layers (PriorLayers' pipeline, generalised to a list
// of layers kept apart from the run's own vectors; the store itself: seg-layers.test.mjs) ----------------------------------
/** Lists layers in PriorLayers' store as loaded ([key, JSON polygons, style]): what seg-layers.mjs holds once their
 *  outlines arrived. Returns the layers. */
function segs(h, specs) {
  const store = h.priors.segments;
  Object.assign(store, { storageGiven: null, sample: 'z1' });
  return specs.map(([key, polys, style = {}]) => {
    const layer = store.make(key, style);
    Object.assign(layer, { set: polygonSetFromLegacy(polys), state: 'ready' });
    store.layers.push(layer);
    return layer;
  });
}
const RUN = 0, SEG = 1;   // world(): the canvases PriorLayers makes, in order
const on = (calls, canvas) => calls.filter(c => c[3] === canvas);
/** A canvas's calls with their paths as subpaths: the same drawing whatever Path2D objects it was made of. */
const drawing = calls => calls.map(([name, args, style]) => [name, args[0] instanceof RecPath ? subpaths(args[0]).map(p => p.join()) : args, style]);
/** Polygons moved into the setup camera's view (x 300-700, y 350-650 µm). */
const inView = polys => polys.map(({ id, poly }) => ({ id, poly: poly.map(([x, y]) => [x + 380, y + 380]) }));
const A = ['xenium_cell_boundaries', grid(30, 30), { color: '#7be29a', opacity: .8, width: 1 }];
const B = ['cellpose_test', inView(grid(12, 12, 15)), { color: '#ffb454', opacity: .5, width: 2, fill: true }];

test('segmentation layers: a raster of their own under the run\'s, one path each (filled, then its one stroke), the first listed on top; the run\'s drawing unchanged', () => {
  const plain = setup(); Object.assign(plain.priors.settings.xenium, XENIUM);
  const alone = plain.draw();
  const h = setup(); Object.assign(h.priors.settings.xenium, XENIUM);
  segs(h, [A, B]);
  const { calls, gl } = h.draw(), z = h.stage.camera.zoom;
  assert.deepEqual(drawing(on(calls, RUN)), drawing(alone.calls), 'the run\'s own canvas: exactly what it was');
  const seg = on(calls, SEG);
  assert.deepEqual(seg.map(c => c[0]), ['setTransform', 'fill', 'stroke', 'stroke'], 'the last listed first (under), filled before its stroke');
  const [, fill, under, over] = seg;
  assert.equal(fill[1][0], under[1][0], 'one path: filled and stroked');
  assert.deepEqual([fill[2].fillStyle, fill[2].globalAlpha], ['#ffb454', .5 * FILL_SHARE]);
  assert.deepEqual([under[2].strokeStyle, under[2].globalAlpha, under[2].lineWidth], ['#ffb454', .5, 2 / z]);
  assert.deepEqual([over[2].strokeStyle, over[2].globalAlpha, over[2].lineWidth], ['#7be29a', .8, 1 / z]);
  const want = new Set(grid(30, 30).map(({ poly }) => poly.flat().map(Math.fround).join()));
  assert.ok(subpaths(over[1][0]).every(p => want.has(p.join())), 'the layer\'s own polygons');
  assert.ok(RecPath.made.flatMap(p => p.closes).every(n => n === 1), 'closePath only on one-polygon paths');
  // Two textures and quads: the segmentation layers' first (under the run's), each its own program.
  assert.deepEqual(named(gl, 'texImage2D').map(c => c[1].at(-1)), [h.w.canvases[SEG], h.w.canvases[RUN]]);
  assert.deepEqual(named(gl, 'useProgram').map(c => c[1][0]), [h.priors.segRaster.p, h.priors.p]);
  assert.equal(named(gl, 'drawArrays').length, 2);
  assert.equal(h.w.measures.at(-2).detail.layer, 'segmentations');
});

test('a segmentation layer\'s change paints and uploads its raster only; a change of the run\'s own layers paints only theirs', () => {
  const h = setup(); Object.assign(h.priors.settings.xenium, XENIUM);
  const [, b] = segs(h, [A, B]);
  h.draw();
  h.stage.dirty = false;
  h.w.with(() => h.priors.segments.setStyle(b, { opacity: .3 }));
  assert.equal(h.stage.dirty, true, 'the store marks the stage dirty');
  let r = h.draw();
  assert.deepEqual([...new Set(r.calls.map(c => c[3]))], [SEG]);
  assert.deepEqual(named(r.gl, 'texImage2D').map(c => c[1].at(-1)), [h.w.canvases[SEG]]);
  assert.equal(named(r.gl, 'drawArrays').length, 2, 'both still drawn');
  assert.equal(strokes(r.calls)[0][2].globalAlpha, .3);
  h.priors.settings.xenium.opacity = .4;
  r = h.draw();
  assert.deepEqual([...new Set(r.calls.map(c => c[3]))], [RUN]);
  assert.deepEqual(named(r.gl, 'texImage2D').map(c => c[1].at(-1)), [h.w.canvases[RUN]]);
  // Hidden, the layer goes from its raster: drawn again without it; with none shown, no texture, no quad of theirs.
  h.w.with(() => h.priors.segments.show(b, false));
  r = h.draw();
  assert.deepEqual(on(r.calls, SEG).map(c => c[0]), ['setTransform', 'stroke']);
  h.w.with(() => h.priors.segments.show(h.priors.segments.layers[0], false));
  r = h.draw();
  assert.deepEqual(on(r.calls, SEG), []); assert.equal(named(r.gl, 'drawArrays').length, 1, 'the run\'s quad only');
  assert.equal(named(r.gl, 'texImage2D').length, 0);
});

test('nothing is drawn while still: no paint, upload or worker request; segLayerDraws counts each new raster, registered at 0', async () => {
  const counts = {}, hook = { count: (name, n = 1) => { counts[name] = (counts[name] || 0) + n; }, priorRebuild() {}, priorWorkerPaint() {} };
  const h = setup({ uploadMs: 120, Layers: await hookedLayers(hook) });
  assert.deepEqual(COUNTERS, ['segLayerDraws']); assert.deepEqual(counts, { segLayerDraws: 0 }, 'registered when the page is made');
  segs(h, [A, B]);
  h.draw();
  assert.equal(counts.segLayerDraws, 1);
  for (let k = 0; k < 20; k++) {
    h.w.advance(1000);
    const r = h.draw();
    assert.deepEqual([r.calls.length, named(r.gl, 'texImage2D').length, named(r.gl, 'drawArrays').length], [0, 0, 1], 'only its quad, as it was');
  }
  assert.equal(counts.segLayerDraws, 1);
  assert.ok(h.w.timers.every(t => t.cleared || t.fired), 'and nothing scheduled');
  // A pan: moved under the camera (no new raster) until the stage settles, then drawn once.
  h.stage.movingUntil = h.w.now + 140; h.stage.camera.x += 10;
  let r = h.draw();
  assert.deepEqual([on(r.calls, SEG).length, named(r.gl, 'texImage2D').length], [0, 0]);
  assert.equal(counts.segLayerDraws, 1);
  h.w.advance(200); r = h.draw();
  assert.deepEqual(on(r.calls, SEG).map(c => c[0]), ['setTransform', 'fill', 'stroke', 'stroke']);
  assert.equal(counts.segLayerDraws, 2);
  h.w.advance(1000); h.draw();
  assert.equal(counts.segLayerDraws, 2, 'settled once');
});

test('segmentation layers are drawn by the worker on every page (a run\'s detail view too): each set sent once, one no longer drawn let go', () => {
  const h = setup({ worker: true }); Object.assign(h.priors.settings.xenium, XENIUM);
  const [a, b] = segs(h, [A, B]);
  let r = h.draw();
  assert.equal(h.w.workers.length, 1, 'the segmentation layers\' worker; the run\'s own outlines stay on the main thread');
  assert.deepEqual([...new Set(r.calls.map(c => c[3]))], [RUN]);
  const worker = h.w.workers[0], first = worker.requests[0], id = first.layers.map(l => l.id);
  assert.deepEqual(first.layers.map(({ id: _, ...style }) => style), [{ color: '#7be29a', opacity: .8, width: 1, fill: false }, { color: '#ffb454', opacity: .5, width: 2, fill: true }]);
  assert.deepEqual(Object.keys(first.sets).map(Number), id);
  assert.deepEqual([first.sets[id[0]].count, first.sets[id[1]].count, 'vectors' in first, first.visible], [900, 144, false, true]);
  assert.equal(named(r.gl, 'texImage2D').length, 1, 'only the run\'s: nothing incomplete is uploaded');
  const image = bitmap(); worker.reply(image, { layer: 'segmentations', polygons: 1044 });
  r = h.draw();
  assert.deepEqual(named(r.gl, 'texImage2D').map(c => c[1].at(-1)), [image]);
  h.stage.camera.x += 5; h.draw(); worker.reply(bitmap()); h.draw();
  assert.equal('sets' in worker.requests[1], false, 'the same sets are not sent again for a new view');
  h.w.with(() => h.priors.segments.show(b, false)); h.draw();
  assert.deepEqual(worker.requests[2].layers.map(l => l.id), [id[0]]); assert.equal('sets' in worker.requests[2], false);
  worker.reply(bitmap()); h.draw();
  h.w.with(() => h.priors.segments.show(b, true)); h.draw();
  assert.deepEqual(Object.keys(worker.requests[3].sets).map(Number), [id[1]], 'shown again: sent again (the worker let it go)');
  assert.equal(worker.requests[3].sets[id[1]].count, 144);
  // All hidden: the worker goes, and its image.
  worker.reply(bitmap()); h.draw();
  h.w.with(() => { h.priors.segments.show(a, false); h.priors.segments.show(b, false); });
  r = h.draw();
  assert.equal(worker.terminated, true); assert.equal(h.priors.segRaster.bitmap, null);
});

test(`at most ${MAX_SHOWN} segmentation layers are drawn; none in 3-D; in the picker by their own eye`, () => {
  const h = setup();
  const five = segs(h, Array.from({ length: 5 }, (_, k) => [`layer_${k}`, inView(grid(4, 4, 10 + k)), { color: '#7be29a' }]));
  assert.equal(five.filter(l => l.style.on).length, 5, 'five on (listed by hand)');
  assert.equal(strokes(on(h.draw().calls, SEG)).length, MAX_SHOWN);
  h.stage.orbit.active = true;
  const r = h.draw();
  assert.deepEqual([r.calls.length, r.gl.length], [0, 0], '3-D: nothing');
  h.stage.orbit.active = false;
  const picker = setup({ info: null });
  const [shown, hidden] = segs(picker, [A, B]);
  picker.w.with(() => picker.priors.segments.show(hidden, false));
  assert.deepEqual(strokes(on(picker.draw().calls, SEG)).map(c => c[2].strokeStyle), [shown.style.color], 'not every layer given, as the prior preview is');
});

test('segmentationsShown: each loaded layer with whether its outlines, as styled now, are in the raster on screen', () => {
  const h = setup();
  assert.deepEqual(h.priors.segmentationsShown(), { sample: null, note: null, loaded: [] });
  const [a, b] = segs(h, [A, B]);
  assert.deepEqual(h.priors.segmentationsShown().loaded.map(l => [l.key, l.visible, l.drawn]), [[A[0], true, false], [B[0], true, false]]);
  h.draw();
  assert.deepEqual(h.priors.segmentationsShown().loaded.map(l => l.drawn), [true, true]);
  h.w.with(() => h.priors.segments.setStyle(a, { color: '#ffffff' }));
  assert.deepEqual(h.priors.segmentationsShown().loaded.map(l => l.drawn), [false, false], 'restyled, not drawn yet');
  h.draw();
  h.w.with(() => h.priors.segments.show(b, false));
  h.draw();
  const shown = h.priors.segmentationsShown();
  assert.deepEqual(shown.loaded.map(l => [l.key, l.visible, l.color, l.drawn]), [[A[0], true, '#ffffff', true], [B[0], false, '#ffb454', false]]);
  assert.deepEqual(JSON.parse(JSON.stringify(shown)), shown, 'plain JSON');
  const bare = Object.create(PriorLayers.prototype);
  assert.deepEqual(bare.segmentationsShown(), { sample: null, note: null, loaded: [] }, 'a page without the store');
});

test('the hook hears each image installed with the cells\' classes (paint time, outlines, kept and hidden cells), never one only drawn', async () => {
  const calls = [], hook = { priorRebuild() {}, priorWorkerPaint() {}, rangePaint: (ms, n, shown) => calls.push([ms, n, shown]) };
  const h = setup({ info: { kind: 'stain' }, worker: true, Layers: await hookedLayers(hook) });
  h.priors.settings.xenium.on = true;
  h.w.with(() => { h.priors.vectors = { cells: grid(2, 1).map((p, i) => ({ ...p, id: i + 1 })), nuclei: null, set: 's' }; });
  h.draw(); const worker = h.w.workers[0]; worker.reply(bitmap(), { paintMs: 5, polygons: 2 }); h.draw();
  assert.deepEqual(calls, [], 'no classes yet: no range paint');
  h.priors.classes = { gen: 1, set: 's', array: new Uint8Array([0, 0]), kept: 2, cells: 2 }; h.draw();
  worker.reply(bitmap(), { paintMs: 6, polygons: 2 });
  assert.deepEqual(calls, [], 'drawn, not on screen yet');
  h.draw(); assert.deepEqual(calls, [[6, 2, { kept: 2, hidden: 0, grey: 0 }]], 'the first all-kept image');
  h.priors.classes = { gen: 2, set: 's', array: new Uint8Array([1, 0]), kept: 1, cells: 2 }; h.draw();
  h.priors.classes = { gen: 3, set: 's', array: new Uint8Array([1, 1]), kept: 0, cells: 2 };
  worker.reply(bitmap(), { paintMs: 7, polygons: 1 }); h.draw();
  assert.deepEqual(calls.at(-1), [7, 1, { kept: 1, hidden: 1, grey: 0 }], 'generation 2 installs while 3 is wanted');
  // An image for a camera that has moved on is never installed, so never counted.
  h.stage.camera.x += 10; h.draw(); h.stage.camera.x += 10; h.draw(); worker.reply(bitmap(), { paintMs: 8, polygons: 0 }); h.draw();
  assert.equal(calls.length, 2);
  worker.reply(bitmap(), { paintMs: 9, polygons: 0 }); h.draw();
  assert.deepEqual(calls.at(-1), [9, 0, { kept: 0, hidden: 2, grey: 0 }]);
  // Classes of another set (the outlines of another cell set, e.g. the run joined its tiles meanwhile) are not this image's.
  const synced = setup({ info: { kind: 'detail' }, camera: { x: 20, y: 10, zoom: 2 }, Layers: await hookedLayers(hook) }), before = calls.length;
  synced.priors.settings.xenium.on = true;
  synced.w.with(() => { synced.priors.vectors = { cells: grid(2, 1).map((p, i) => ({ ...p, id: i + 1 })), nuclei: null, set: 'joined' }; });
  synced.priors.classes = { gen: 1, set: 'tiles', array: new Uint8Array([1, 1]), kept: 0, cells: 2 }; synced.draw();
  assert.equal(calls.length, before);
  synced.priors.classes = { gen: 2, set: 'joined', array: new Uint8Array([1, 0]), kept: 1, cells: 2 }; synced.draw();
  assert.deepEqual(calls.at(-1).slice(1), [1, { kept: 1, hidden: 1, grey: 0 }], 'the synchronous renderer reports its own paint');
});
