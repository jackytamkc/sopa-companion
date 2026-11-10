import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { TiledLoader } from '../src/tiled-loader.mjs';
import { Orbit } from '../src/scene3d.js';
import { domFixture } from './dom-fixture.mjs';

// "The 3D function is broken, always glitch back to 2D": a tiled run (a whole sample, or a region over 1,200 µm) in
// 3-D, wired as main.js wires it.
// The real TiledLoader, the real Orbit, the Stage's own camera code (stage.js, from its pixel ratio to emit()) and
// main.js's own tiledStatus, setView and segmented; the engine is faked as server/app.py (raster_window: the view is
// clamped to proseg's grid) and proseg-live's live_window_rect (the paused window: the posted box, clamped and capped).
// The geometry is a real run's: a 1,152 x 1,260 µm box, 1 µm voxels, window_max 1,600, 4 z layers,
// held at the end of its schedule (seq 451), on a stage of 1559 x 1040 device px.

const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const stageSource = readFileSync(new URL('../src/stage.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const between = (source, from, to) => {
  const a = source.indexOf(from), b = source.indexOf(to, a);
  assert.ok(a >= 0 && b > a, `${from.trim()} … ${to.trim()}`);
  return source.slice(a, b);
};
const camera = vm.createContext({ performance });
vm.runInContext('this.CameraStage = class {' + between(stageSource, '  /** Device px per CSS px (cached by resize). */',
  '  /** The label under a world point in the shown frame (0 = none). */') + '}', camera);
const settle = () => new Promise(resolve => setImmediate(resolve));
const NOTCH = Math.exp(100 * 0.0016);   // one wheel notch (main.js setupPointer: deltaY 100)

function page(t, { box = [0, 0, 1152, 1260], origin = [0, 0], runState = 'paused' } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const realMatch = globalThis.matchMedia;
  globalThis.matchMedia = () => ({ matches: true });   // reduced motion: the 2-D/3-D switch is at once
  t.after(() => { globalThis.matchMedia = realMatch; });
  // The page: index.html's view switch, z layers row, hint and tooltip.
  const dom = domFixture(); dom.document.body.innerHTML = html;
  const $ = dom.$, button3d = () => $('view').children.find(b => b.dataset.value === '3d');
  $('view').querySelector = () => button3d();
  // proseg's grid: from the first transcript (origin), whole 150 µm quads to the right and below.
  const voxel = 1, grid = { x0: origin[0], y0: origin[1], nx: 1200, ny: 1350, nz: 4 };
  const run = { id: 'r', mode: 'tiled', state: runState, box, info: { window_max: 1600, burnin_voxel_size: 2, layers: 4 }, image: null, window: null };
  const header = { seq: 451, iter: 450, format: 'tiles', x0: grid.x0, y0: grid.y0, nx: grid.nx, ny: grid.ny, nz: 4, voxel, window: null, view: null };
  // The stage: the Stage's camera methods, its canvas, and the real orbit.
  const stage = new camera.CameraStage();
  Object.assign(stage, { canvas: { width: 1559, height: 1040, clientWidth: 1559, clientHeight: 1040 }, pixelRatio: 1,
    camera: { x: 576, y: 630, zoom: 3 }, listeners: new Set(), options: {}, frame: null, wake() {}, set() {}, setTranscripts() {} });
  stage.orbit = new Orbit(stage);
  // The engine.
  let windows = 0;
  const posted = [];
  const windowFor = view => {   // proseg-live live_window_rect + refresh_window's window.json
    const max = run.info.window_max;
    const clamp = (a, b, n) => {
      a = Math.min(Math.max(a, 0), n); b = Math.min(Math.max(b + 1, 0), n);
      if (b - a > max) { const centre = Math.floor((a + b) / 2); a = Math.min(Math.max(centre - Math.floor(max / 2), 0), n); b = Math.min(a + max, n); }
      return [a, b];
    };
    const [i0, i1] = clamp(Math.floor((view[0] - grid.x0) / voxel), Math.floor((view[2] - grid.x0) / voxel), grid.nx);
    const [j0, j1] = clamp(Math.floor((view[1] - grid.y0) / voxel), Math.floor((view[3] - grid.y0) / voxel), grid.ny);
    return { seq: header.seq, iter: header.iter, i0, j0, nx: i1 - i0, ny: j1 - j0, nz: grid.nz, raster_nx: grid.nx, raster_ny: grid.ny,
      x0: grid.x0, y0: grid.y0, voxel, file: `window-${++windows}.bin.zst`, view };
  };
  const packed = (name, meta, data) => new Response(data.buffer, { headers: { [name]: JSON.stringify(meta) } });
  t.mock.method(globalThis, 'fetch', async (url, options = {}) => {
    const u = new URL(url, 'http://test'), p = u.pathname;
    if (options.method === 'POST') { posted.push(JSON.parse(options.body).box); return new Response(JSON.stringify({ view: posted.at(-1) })); }
    if (p.endsWith('/xenium')) return new Response(JSON.stringify({ detail: 'no prior vectors here' }), { status: 404 });
    if (p.includes('/transcripts/view')) return new Response(new ArrayBuffer(0));
    if (p.endsWith('/window')) {
      const w = run.window;
      if (!w) return new Response(JSON.stringify({ detail: 'No window in this frame' }), { status: 404 });
      return packed('X-Window', { seq: w.seq, iter: w.iter, x0: w.x0 + w.i0 * voxel, y0: w.y0 + w.j0 * voxel, nx: w.nx, ny: w.ny, nz: w.nz, voxel, source: 'paused' },
        new Uint32Array(w.nx * w.ny * w.nz).fill(7));
    }
    const b = u.searchParams.get('box').split(',').map(Number);   // server/app.py raster_window (stride 1)
    const c0 = Math.max(0, Math.floor((b[0] - grid.x0) / voxel)), r0 = Math.max(0, Math.floor((b[1] - grid.y0) / voxel));
    const c1 = Math.min(grid.nx, Math.ceil((b[2] - grid.x0) / voxel)), r1 = Math.min(grid.ny, Math.ceil((b[3] - grid.y0) / voxel));
    const nx = c1 - c0, ny = r1 - r0;
    return packed('X-View', { seq: header.seq, iter: header.iter, x0: grid.x0 + c0 * voxel, y0: grid.y0 + r0 * voxel, nx, ny, stride: 1, voxel,
      planes: ['consensus', 'last_change'] }, new Uint32Array(nx * ny * 2).fill(1));
  });
  // main.js: tiledStatus, setView and segmented as they are, and the view switch wired as setupControls wires it.
  const state = { run, display: { layer: -1 }, hover: 0 };
  const c = vm.createContext({ $, stage, state, fmt1: new Intl.NumberFormat('en-GB', { maximumFractionDigits: 1 }), drawOverlay() {} });
  vm.runInContext(between(main, 'function tiledStatus(', 'function phaseLabel(') + between(main, 'function setView(', '/** Cells (filled, key 1)') +
    between(main, 'function segmented(', 'let noticeTimer'), c);
  c.segmented($('view'), value => c.setView(value === '3d'));
  // main.js show() for a tiled run (the parts the loader sees), refreshTiled, the camera listener and the run event.
  const shows = [], errors = [];
  const show = async () => {
    const data = await loader.frame(header);
    if (!data) return;
    stage.frame = { nx: data.nx, ny: data.ny, x0: data.x0, y0: data.y0, voxel: data.voxel, layers: data.layers };
    loader.committed(header, data);
  };
  const loader = new TiledLoader(stage, { setRaster() {} }, { refresh: () => { shows.push(show()); }, status: s => c.tiledStatus(s),
    error: e => errors.push(e), image() {}, channels: () => [] });
  stage.onChange(() => loader.move());
  t.after(() => loader.reset());
  const idle = async () => { for (let k = 0; k < 6; k++) { await settle(); while (shows.length) await shows.shift(); } };
  return {
    run, stage, loader, header, posted, errors, $, button3d,
    in3d: () => stage.orbit.enabled,
    async open() { loader.reset(run); shows.push(show()); await idle(); },
    /** The view's window: posted after 300 ms, written by proseg, read by the engine and pushed (a run event). */
    async windowArrives() {
      t.mock.timers.tick(300); await idle();
      assert.ok(posted.length, 'the view was posted');
      run.window = windowFor(posted.at(-1));
      loader.run = run;                                                   // main.js onMessage 'run' (tiled)
      if (loader.windowChanged(header)) shows.push(show());
      await idle();
    },
    async moved() { t.mock.timers.tick(120); await idle(); },          // the camera came to rest: refreshTiled
    click3d() { $('view').emit('click', { target: button3d() }); },
  };
}

for (const [gesture, move] of [
  ['a right-drag pan of 200 px', p => p.stage.orbit.drag(0, 200, true)],
  ['two wheel notches toward the far side of the tilted view', p => { p.stage.zoomAt(780, 60, NOTCH); p.stage.zoomAt(780, 60, NOTCH); }],
]) test(`a tiled run stays in 3-D after ${gesture} (its layers follow the view a moment later)`, async t => {
  const p = page(t);
  await p.open();
  assert.equal(p.button3d().disabled, true, 'no layers before the first window');
  await p.windowArrives();
  assert.equal(p.stage.frame.layers.length, 4); assert.equal(p.button3d().disabled, false);
  p.click3d();
  assert.equal(p.in3d(), true, '3-D on');
  await p.moved();
  assert.equal(p.in3d(), true, 'turning to 3-D alone keeps it (the orbit does not move the 2-D camera)');
  move(p);
  assert.equal(p.in3d(), true, `3-D is kept while the layers follow the view (${gesture}); it went back to 2-D at once`);
  await p.moved();
  assert.equal(p.in3d(), true, 'and while the new view waits for its window');
  await p.windowArrives();
  assert.equal(p.stage.frame.layers.length, 4, 'the new view has its layers');
  assert.equal(p.in3d(), true, 'still in 3-D once they are in');
  assert.equal(p.button3d().disabled, false); assert.equal(p.$('tiled-hint').hidden, true);
});

test('a tiled run still returns to 2-D where no layers can come: zoomed out past window_max', async t => {
  const p = page(t);
  await p.open(); await p.windowArrives(); p.click3d(); await p.moved();
  assert.equal(p.in3d(), true);
  p.stage.camera.zoom = 0.9; p.stage.emit();                 // 1,732 µm across
  assert.equal(p.in3d(), false); assert.equal(p.button3d().title, 'Zoom in to ≤ 1.6 mm to see this part in 3-D');
});

test('a finished tiled run still returns to 2-D at a part it never showed while it ran', async t => {
  const p = page(t, { runState: 'done' });
  await p.open(); await p.windowArrives(); p.click3d(); await p.moved();
  assert.equal(p.in3d(), true);
  p.stage.orbit.drag(0, 200, true);
  assert.equal(p.in3d(), false); assert.equal(p.button3d().title, 'This finished run has 3-D layers only where it was viewed');
});

// A second way to lose 3-D: proseg's grid starts at the run's first transcript, inside the drawn box (on every drawn box
// in the harness runs: by 0.002-5.5 µm; a dense run on the same sample: 0.0135 µm in x, 0.0029
// µm in y; a tiled run: 26 µm and 306 µm), and the window cannot reach past the grid. A view whose box reaches the
// run box's left or top edge then never gets layers, while its hint says they follow.
test('a view at the region\'s top-left edge gets its 3-D layers although the grid starts at the first transcript', async t => {
  const p = page(t, { origin: [0.0135, 0.0029] });
  p.stage.camera.x = 300; p.stage.camera.y = 220;            // the region's top-left part, at the same zoom
  await p.open(); await p.windowArrives();
  assert.equal(p.posted.at(-1)[0], 0, 'the view box starts at the run box'); assert.equal(p.posted.at(-1)[1], 0);
  assert.equal(p.stage.frame.layers.length, 4, 'the window covers every column of the view that has voxels');
  assert.equal(p.button3d().disabled, false);
});
