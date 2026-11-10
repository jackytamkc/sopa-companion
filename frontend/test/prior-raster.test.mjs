import test from 'node:test';
import assert from 'node:assert/strict';
import { PriorRaster } from '../src/prior-raster.mjs';
import { layerPath, FILL_SHARE } from '../src/prior-paths.mjs';
import { polygonSetFromLegacy } from '../src/polygons.mjs';

function environment(t) {
  const strokes = [], snapshots = [], contexts = [], canvases = [], paths = [];
  class Path {
    constructor() { this.ops = []; paths.push(this); }
    moveTo(...p) { this.ops.push(['M', ...p]); }
    lineTo(...p) { this.ops.push(['L', ...p]); }
    closePath() { this.ops.push(['Z']); }
    addPath(p) { this.ops.push(...p.ops); }
  }
  const globals = {
    Path2D: Path,
    OffscreenCanvas: class {
      constructor(width, height) {
        Object.assign(this, { width, height }); canvases.push(this);
        const g = this.g = {
          setTransform(...matrix) { g.matrix = matrix; },
          stroke(path) { strokes.push({ ops: path.ops, color: g.strokeStyle, alpha: g.globalAlpha, width: g.lineWidth, matrix: g.matrix }); },
          fill(path) { strokes.push({ fill: true, ops: path.ops, color: g.fillStyle, alpha: g.globalAlpha }); },
        };
      }
      getContext(...args) { contexts.push(args); return this.g; }
    },
    createImageBitmap: async (source, options) => {
      const bitmap = { width: source.width, height: source.height, close() { this.closed = true; } };
      snapshots.push({ source, options, bitmap }); return bitmap;
    },
  };
  const saved = Object.fromEntries(Object.keys(globals).map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  for (const [k, value] of Object.entries(globals)) Object.defineProperty(globalThis, k, { configurable: true, writable: true, value });
  t.after(() => { for (const [k, descriptor] of Object.entries(saved)) { if (descriptor) Object.defineProperty(globalThis, k, descriptor); else delete globalThis[k]; } });
  return { strokes, snapshots, contexts, canvases, paths, Path };
}
const rectangle = (x, y, w, h) => ({ poly: [[x, y], [x + w, y], [x + w, y + h], [x, y + h]] });
function request() {
  return {
    id: 1, view: { x: 10.5, y: 20.25, zoom: 2, width: 1854, height: 965, ratio: 2 },
    settings: { xenium: { on: false, width: 1, color: '#8ecae6', opacity: .7 }, nuclei: { on: false, width: 2.2, color: '#ffd18a', opacity: .8 } },
    vectors: { cells: polygonSetFromLegacy([rectangle(0, 0, 20, 20), rectangle(20, 0, 20, 20)]), nuclei: polygonSetFromLegacy([rectangle(5, 5, 5, 5)]) },
    moving: false, visible: true,
  };
}

test('worker retains classes across views, binds them to the set and hides the layer when switched off',async t=>{
  const h=environment(t),raster=new PriorRaster(),r=request();
  r.vectors.set='s';r.vectors.nuclei=null;r.vectors.cells.ids=new Uint32Array([1,2]);
  r.classes={gen:1,set:'s',array:new Uint8Array([1,0])};
  let result=await raster.draw(r);assert.equal(result.polygons,1);assert.equal(h.strokes.length,1);assert.equal(h.strokes[0].fill,undefined);
  const {vectors,classes,...next}=r;result=await raster.draw(next);assert.equal(result.polygons,1);
  result=await raster.draw({...next,classes:{gen:2,set:'different',array:new Uint8Array([1,1])}});assert.equal(result.polygons,2);
  result=await raster.draw({...next,picker:false});assert.equal(result.polygons,0,'stain layers respect the eye switch');
  result=await raster.draw({...next,classes:null});assert.equal(result.polygons,2);
});

test('the worker uses the unchanged union strokes and prepares a straight-alpha bitmap without an extra canvas copy', async t => {
  const h = environment(t), raster = new PriorRaster(), r = request();
  const result = await raster.draw(r);
  assert.deepEqual(h.contexts, [['2d']], 'default 2-D context, no alternate rasteriser options');
  assert.equal(h.strokes.length, 2, 'one union stroke per layer even though run settings are off in the picker');
  assert.deepEqual(h.strokes[0], {
    ops: [['M', 0, 0], ['L', 20, 0], ['L', 20, 20], ['L', 0, 20], ['Z'], ['M', 20, 0], ['L', 40, 0], ['L', 40, 20], ['L', 20, 20], ['Z']],
    color: '#8ecae6', alpha: .7, width: 1, matrix: [2, 0, 0, 2, 906, 442],
  });
  assert.deepEqual(h.strokes[1], {
    ops: [['M', 5, 5], ['L', 10, 5], ['L', 10, 10], ['L', 5, 10], ['Z']],
    color: '#ffd18a', alpha: .8, width: 2.2, matrix: [2, 0, 0, 2, 906, 442],
  });
  assert.equal(h.canvases.length, 1);
  assert.equal(h.snapshots[0].source, raster.canvas);
  assert.deepEqual(h.snapshots[0].options, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  assert.equal(result.bitmap, h.snapshots[0].bitmap);
  assert.deepEqual([result.bitmap.width, result.bitmap.height, result.polygons, result.layer], [1854, 965, 3, 'xenium+nuclei']);
});

test('worker views reuse their vectors and canvas; replacing vectors releases old path tiles', async t => {
  const h = environment(t), raster = new PriorRaster(), r = request();
  await raster.draw(r);
  const before = h.paths.length, { vectors, ...next } = r;
  next.view = { ...r.view, width: 1000, ratio: 1 };
  await raster.draw(next);
  assert.equal(raster.vectors, vectors);
  assert.equal(h.canvases.length, 1, 'a long-lived canvas, resized just as the main-thread renderer was');
  assert.equal(h.paths.length - before, 2, 'only the view paths are new; tiles are reused');
  assert.deepEqual([h.strokes.at(-1).width, h.snapshots.at(-1).bitmap.width], [1.1, 1000]);
  await raster.draw({ ...r, vectors: { cells: null, nuclei: null } });
  const released = h.paths.length;
  layerPath(vectors.cells, [-100, -100, 100, 100], 2, h.Path);
  assert.ok(h.paths.length - released > 1, 'replaced vectors no longer own cached tile paths');
});

test('a worker result waits for bitmap completion and propagates snapshot failures', async t => {
  environment(t);
  let complete;
  globalThis.createImageBitmap = () => new Promise(resolve => { complete = resolve; });
  const raster = new PriorRaster();
  let finished = false;
  const pending = raster.draw(request()).then(r => { finished = true; return r; });
  await Promise.resolve();
  assert.equal(finished, false);
  const bitmap = {}; complete(bitmap);
  assert.equal((await pending).bitmap, bitmap);
  globalThis.createImageBitmap = async () => { throw Error('snapshot failed'); };
  await assert.rejects(raster.draw(request()), /snapshot failed/);
});

test('the worker transfers the completed bitmap and returns an error with the request id on failure', async t => {
  environment(t);
  const sent = [], old = Object.getOwnPropertyDescriptor(globalThis, 'self');
  const worker = globalThis.self = { postMessage: (...args) => sent.push(args) };
  t.after(() => { if (old) Object.defineProperty(globalThis, 'self', old); else delete globalThis.self; });
  await import('../src/prior-raster-worker.mjs');
  await worker.onmessage({ data: request() });
  const [result, transfer] = sent[0];
  assert.equal(result.id, 1);
  assert.deepEqual(transfer, [result.bitmap], 'ownership, not a second image copy');
  globalThis.createImageBitmap = async () => { throw Error('snapshot failed'); };
  await worker.onmessage({ data: { ...request(), id: 2 } });
  assert.deepEqual(sent[1], [{ id: 2, error: 'snapshot failed' }]);
  const bitmap = { close() { this.closed = true; } };
  globalThis.createImageBitmap = async () => bitmap;
  worker.postMessage = (...args) => { if (args[1]) throw Error('transfer failed'); sent.push(args); };
  await worker.onmessage({ data: { ...request(), id: 3 } });
  assert.equal(bitmap.closed, true, 'a bitmap that could not be transferred is released');
  assert.deepEqual(sent[2], [{ id: 3, error: 'transfer failed' }]);
});

test('the segmentation layers\' worker: each set comes once by id and is kept while listed; one no longer listed is let go', async t => {
  const h = environment(t), raster = new PriorRaster(), view = request().view;
  const a = polygonSetFromLegacy([rectangle(0, 0, 20, 20)]), b = polygonSetFromLegacy([rectangle(5, 5, 5, 5), rectangle(30, 0, 5, 5)]);
  const style = { color: '#7be29a', opacity: .8, width: 1, fill: false };
  let result = await raster.draw({ id: 1, view, visible: true, moving: false, sets: { 7: a, 9: b },
    layers: [{ id: 7, ...style }, { id: 9, color: '#ffb454', opacity: .5, width: 2, fill: true }] });
  assert.deepEqual(h.strokes.map(s => [s.fill ? 'fill' : 'stroke', s.color, s.alpha]), [['fill', '#ffb454', .5 * FILL_SHARE], ['stroke', '#ffb454', .5], ['stroke', '#7be29a', .8]],
    'the last listed under the first; filled, then stroked');
  assert.deepEqual([result.polygons, result.layer], [3, 'segmentations']);
  assert.equal(raster.vectors, null, 'no run vectors in this worker');
  h.strokes.length = 0;
  result = await raster.draw({ id: 2, view, visible: true, moving: false, layers: [{ id: 7, ...style }, { id: 9, ...style }] });
  assert.equal(h.strokes.length, 2, 'kept: drawn without being sent again');
  assert.deepEqual([...raster.sets.keys()], [7, 9]);
  const before = h.paths.length;
  layerPath(b, [-100, -100, 100, 100], 2, h.Path);
  assert.equal(h.paths.length - before, 1, 'its tiles are kept with it');
  await raster.draw({ id: 3, view, visible: true, moving: false, layers: [{ id: 7, ...style }] });
  assert.deepEqual([...raster.sets.keys()], [7]);
  const released = h.paths.length;
  layerPath(b, [-100, -100, 100, 100], 2, h.Path);
  assert.ok(h.paths.length - released > 1, 'one no longer listed is let go with its tiles');
});
