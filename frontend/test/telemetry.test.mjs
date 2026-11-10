import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { Telemetry, percentile, fetchKind, post, scriptPath, invokerName, measureName, measureDetail, PERIOD_MS, ENDPOINT, MAX_BYTES, STRETCH, VERSION, IDLE_FRAMES, WATCH } from '../src/telemetry.mjs';
import { fakeLoop } from './fake-loop.mjs';

test('percentiles use the nearest rank; fetches are classified by what they load', () => {
  assert.equal(percentile([], .95), null);
  assert.equal(percentile([5], .95), 5);
  assert.equal(percentile(Array.from({ length: 100 }, (_, i) => 100 - i), .95), 95);
  assert.equal(percentile([1, 2, 3, 4], .5), 2);
  const base = 'http://127.0.0.1:8777/api/v1/runs/r1';
  assert.equal(fetchKind(`${base}/frames/12/view?box=0,0,1,1&stride=2&max_px=10`), 'view');
  assert.equal(fetchKind(`${base}/frames/12/window`), 'view');
  assert.equal(fetchKind(`${base}/prior/view?box=1,2,3,4`), 'view');
  assert.equal(fetchKind('/api/v1/runs/r1/frames/7'), 'frame');
  assert.equal(fetchKind(`${base}/image/view/c0.u16?box=0,0,1,1&px=900`), 'image');
  assert.equal(fetchKind(`${base}/image/c1.u16`), 'image');
  assert.equal(fetchKind(`${base}/transcripts/view?box=0,0,1,1`), 'transcripts');
  assert.equal(fetchKind(`${base}/transcripts`), 'transcripts');
  // The prior shapes of a run (JSON, or the binary of format=bin) and the picker's outlines: items 2 and 3 of the
  // 2026-10-17 baseline (75.6 MB of a whole sample's vectors, up to 92 MB of picker outlines).
  assert.equal(fetchKind(`${base}/xenium`), 'vectors');
  assert.equal(fetchKind(`${base}/xenium?format=bin&v=1`), 'vectors');
  // Xenium's label masks are rasters of their own, not prior outlines.
  assert.equal(fetchKind(`${base}/xenium/masks.bin`), 'masks');
  assert.equal(fetchKind(`${base}/xenium/masks`), 'masks');
  assert.equal(fetchKind('/api/v1/samples/sample-1/shapes/xenium_cells?box=0,0,1,1&limit=20000'), 'shapes');
  // A sample's segmentation layers (their list and outlines) and Compare's status and points.
  assert.equal(fetchKind('/api/v1/samples/sample-2/segmentations'), 'segmentations');
  assert.equal(fetchKind('/api/v1/samples/sample-2/segmentations/cellpose_test/outlines?v=0123456789ab'), 'segmentations');
  assert.equal(fetchKind(`${base}/compare`), 'compare');
  assert.equal(fetchKind(`${base}/compare/points/2.bin`), 'compare');
  for (const other of [`${base}/cellstats/3`, `${base}/analysis/4/cells`, '/api/v1/runs', 'http://example.com/frames/1/view', `${base}/frames`, '/src/xenium-masks.js'])
    assert.equal(fetchKind(other), null, other);
});

function telemetry(context = () => ({ mode: 'tiled', state: 'running', cells: 40000, live: true, umap: true, view: '2d', level: 'overview', stage: [1500, 900], renderer: 'ANGLE (NVIDIA Example GPU)' })) {
  let clock = 0;
  const sent = [], counts = { stage: 0, umap: 0, timeline: 0, batches: 0, snapshots: 0 };
  const t = new Telemetry({ context, counters: () => ({ ...counts }), send: body => sent.push(JSON.parse(body)), now: () => clock });
  const win = { devicePixelRatio: 2, innerWidth: 1920, innerHeight: 1080, screen: { width: 2560, height: 1440 }, document: { hidden: false }, performance: { memory: { usedJSHeapSize: 300 * 2 ** 20 } } };
  return { t, sent, counts, win, advance: ms => { clock += ms; } };
}

test('a period summary: rates, frames, frame lag, long tasks, input delay, memory, display, fetches; renderer once', () => {
  const h = telemetry();
  h.advance(10000);
  Object.assign(h.counts, { stage: 22, umap: 3, timeline: 20, batches: 21, snapshots: 20 });
  for (let k = 0; k < 100; k++) h.t.frame(1000 + 100 * k, 1000 + 100 * k + (k < 95 ? 10 : 200));
  h.t.longTask(80); h.t.longTask(230);
  for (let k = 1; k <= 20; k++) h.t.input(k, 10 * k);
  h.t.fetched('view', 1e6, 19e6); h.t.fetched('view', 1e6, 19e6); h.t.fetched('image', 2e6, 4e6); h.t.fetched(null, 5, 5);
  h.t.tick(h.win);
  assert.equal(h.sent.length, 1);
  const s = h.sent[0];
  assert.equal(s.v, VERSION); assert.equal(VERSION, 2);
  assert.deepEqual(s.run, { mode: 'tiled', state: 'running', cells: 40000, live: true, umap: true, view: '2d', level: 'overview', stage: [1500, 900] });
  assert.equal(s.period_s, 10); assert.equal(s.hidden, false);
  // Without input nothing was watched: the frames are those of the busiest surface (a snapshot's panels share its frame).
  assert.deepEqual(s.rates, { stage: 2.2, umap: 0.3, timeline: 2, batches: 2.1, snapshots: 2, frames: 2.2 });
  assert.deepEqual(s.raf_lag_ms, { p95: 10, max: 200 });
  assert.deepEqual(s.long_tasks, { count: 2, max_ms: 230 });
  assert.deepEqual(s.input, { count: 20, delay_p95_ms: 19, delay_max_ms: 20, duration_p95_ms: 190, duration_max_ms: 200 });
  assert.equal(s.heap_mb, 300);
  assert.deepEqual(s.display, { dpr: 2, window: [1920, 1080], screen: [2560, 1440], renderer: 'ANGLE (NVIDIA Example GPU)' });
  assert.deepEqual(s.fetch, { view: { n: 2, bytes: 38e6, wire: 2e6 }, image: { n: 1, bytes: 4e6, wire: 2e6 } });
  assert.deepEqual(s.measures, {});
  assert.equal('raf_frames' in s.rates || 'raf_callbacks' in s.rates, false, 'no rAF wrapper counts any more');
  // The next period starts from zero and names the renderer no more.
  h.advance(5000); h.counts.stage += 5; h.t.tick(h.win);
  const next = h.sent[1];
  assert.equal(next.rates.stage, 1); assert.equal(next.rates.frames, 1); assert.equal(next.display.renderer, undefined); assert.deepEqual(next.fetch, {});
  assert.deepEqual(next.long_tasks, { count: 0, max_ms: 0 }); assert.deepEqual(next.raf_lag_ms, { p95: null, max: null });
  assert.doesNotMatch(JSON.stringify(h.sent), /cell_id|path|zarr|sample/i);
});

test('nothing is sent while no run is open, a restarted counter counts from zero, and nothing ever throws', () => {
  let open = false;
  const h = telemetry(() => open ? { mode: 'detail', state: 'paused', cells: 900, live: false, umap: false, view: '3d', level: null, stage: [800, 600], renderer: null } : null);
  h.advance(10000); h.counts.stage = 50; h.t.tick(h.win);
  assert.equal(h.sent.length, 0);
  open = true; h.advance(10000); h.counts.stage = 70; h.counts.umap = 4; h.t.tick(h.win);
  assert.equal(h.sent[0].rates.stage, 2, 'the closed period was not carried over'); assert.equal(h.sent[0].rates.umap, .4);
  assert.equal(h.sent[0].rates.frames, 2);
  h.advance(10000); h.counts.umap = 1; h.t.tick(h.win);   // a new UMAP view started counting again
  assert.equal(h.sent[1].rates.umap, .1); assert.equal(h.sent[1].rates.frames, .1);
  assert.equal(h.sent[0].heap_mb, 300); assert.equal('renderer' in h.sent[0].display, false);
  const broken = new Telemetry({ context: () => { throw new Error('page torn down'); }, send: () => { throw new Error('offline'); } });
  assert.doesNotThrow(() => broken.tick({}));
  const offline = new Telemetry({ context: () => ({}), send: () => { throw new Error('offline'); } });
  assert.doesNotThrow(() => offline.tick({}));
  let torn = false;
  const failing = new Telemetry({ context: () => ({}), counters: () => { if (torn) throw new Error('torn down'); return {}; } });
  failing.win = { requestAnimationFrame: cb => cb(0) }; failing.rendered = () => failing.check();
  torn = true;
  assert.doesNotThrow(() => { failing.watch(); failing.drew(true); failing.tick({}); });
  assert.equal(failing.watching, false, 'a page whose counters fail draws nothing: the watch ends');
});

test('start() leaves requestAnimationFrame alone; observes long tasks, input, resources, measures; ticks every 10 s; stop() undoes it', () => {
  const observers = [], timers = [], listeners = new Map();
  let clock = 0;
  const callbacks = [];
  const win = {
    requestAnimationFrame(cb) { callbacks.push(cb); return callbacks.length; }, cancelAnimationFrame(id) { win.cancelled = id; },
    PerformanceObserver: class {
      static supportedEntryTypes = ['longtask', 'event', 'resource', 'measure'];
      constructor(fn) { this.fn = fn; observers.push(this); }
      observe(options) { if (options.type === 'longtask' && win.noLongTasks) throw new TypeError('unsupported'); this.options = options; }
      disconnect() { this.disconnected = true; }
      emit(entries) { this.fn({ getEntries: () => entries }); }
    },
    addEventListener(type, fn, options) { listeners.set(type, { fn, options }); }, removeEventListener(type, fn) { if (listeners.get(type)?.fn === fn) listeners.delete(type); },
    setInterval(fn, ms) { timers.push({ fn, ms }); return timers.length; }, clearInterval(id) { timers[id - 1].cleared = true; },
    performance: { clearResourceTimings() { win.cleared = (win.cleared || 0) + 1; } }, document: { hidden: true },
  };
  const native = win.requestAnimationFrame, sent = [], counts = { stage: 0 };
  const t = new Telemetry({ context: () => ({ mode: 'detail' }), counters: () => ({ ...counts }), send: body => sent.push(JSON.parse(body)), now: () => clock }).start(win);
  assert.equal(win.requestAnimationFrame, native, 'not wrapped: long animation frames name the page\'s own callbacks');
  assert.equal(callbacks.length, 0, 'a page nobody touches runs no frame callback');
  const [long, event, resource, measure] = observers;
  assert.deepEqual([long.options.type, event.options.type, event.options.durationThreshold, resource.options.type, measure.options.type, measure.options.buffered],
    ['longtask', 'event', 16, 'resource', 'measure', true]);
  assert.deepEqual([...listeners.keys()], WATCH);
  assert.ok([...listeners.values()].every(l => l.options.capture && l.options.passive), 'passive, and before the page\'s own handlers');
  long.emit([{ duration: 120 }]);
  event.emit([{ startTime: 100, processingStart: 160, duration: 240 }]);
  resource.emit([{ name: 'http://127.0.0.1:8777/api/v1/runs/r/frames/3/view?box=1', encodedBodySize: 1000, decodedBodySize: 9000 }, { name: 'http://127.0.0.1:8777/api/v1/runs', encodedBodySize: 50, decodedBodySize: 50 }]);
  measure.emit([{ name: 'plv:prior-rebuild', duration: 25890.4, detail: { polygons: 41190, layer: 'cells' } }, { name: 'next-render', duration: 3 }]);
  // Input starts a frame watch (here without a MessageChannel: checked in the callback itself).
  clock = 5; listeners.get('pointermove').fn(); listeners.get('wheel').fn();
  assert.equal(callbacks.length, 1, 'one watch at a time');
  clock = 21; counts.stage++; callbacks[0](21);
  assert.equal(callbacks.length, 2); assert.deepEqual(t.raf.lags, [16]); assert.equal(t.raf.frames, 1);
  assert.equal(timers.length, 1); assert.equal(timers[0].ms, PERIOD_MS);
  clock = 10047; timers[0].fn();
  assert.equal(sent.length, 1); assert.equal(win.cleared, 1);
  assert.equal(sent[0].hidden, true); assert.deepEqual(sent[0].long_tasks, { count: 1, max_ms: 120 });
  assert.equal(sent[0].input.delay_max_ms, 60); assert.equal(sent[0].input.duration_max_ms, 240);
  assert.deepEqual(sent[0].fetch, { view: { n: 1, bytes: 9000, wire: 1000 } });
  assert.deepEqual(sent[0].measures, { 'plv:prior-rebuild': { n: 1, max_ms: 25890.4, total_ms: 25890.4, last_ms: 25890.4, last: { polygons: 41190, layer: 'cells' } } });
  assert.equal(sent[0].rates.frames, .1); assert.deepEqual(sent[0].raf_lag_ms, { p95: 16, max: 16 });
  t.stop();
  assert.equal(win.requestAnimationFrame, native); assert.ok(observers.every(o => o.disconnected)); assert.ok(timers[0].cleared);
  assert.equal(listeners.size, 0); assert.equal(win.cancelled, 2, 'the pending watch frame is cancelled'); assert.equal(t.watching, false);
  // A browser without one of the entry types still reports the others.
  win.noLongTasks = true; observers.length = 0;
  const partial = new Telemetry({ context: () => null }).start(win);
  assert.deepEqual(observers.filter(o => o.options).map(o => o.options.type), ['event', 'resource', 'measure']);
  partial.stop();
});

test('no module of the page wraps requestAnimationFrame: a wrapper would be the script of every long animation frame', () => {
  const src = new URL('../src/', import.meta.url);
  // Any way of replacing it: an assignment (also through brackets or a string key), defineProperty or
  // Object.assign/Reflect.set onto window or globalThis with that name.
  const replacing = [/\brequestAnimationFrame['"`]?\s*\]?\s*=(?!=)/, /defineProperty\s*\([^)]*['"`]requestAnimationFrame['"`]/,
    /Reflect\.set\s*\([^)]*['"`]requestAnimationFrame['"`]/, /Object\.assign\s*\(\s*(?:window|globalThis|self)\b[^)]*\brequestAnimationFrame\s*:/];
  for (const name of readdirSync(src).filter(n => /\.m?js$/.test(n))) {
    const code = readFileSync(new URL(name, src), 'utf8').replace(/\/\/.*$/gm, '');
    for (const rx of replacing) assert.doesNotMatch(code, rx, `${name}: ${rx}`);
  }
  for (const wrapper of ["window['requestAnimationFrame'] = f;", 'window["requestAnimationFrame"]=f', 'globalThis.requestAnimationFrame = f',
    "Object.defineProperty(window, 'requestAnimationFrame', { value: f })", "Reflect.set(window, 'requestAnimationFrame', f)",
    'Object.assign(window, { requestAnimationFrame: f })'])
    assert.ok(replacing.some(rx => rx.test(wrapper)), `the guard catches ${wrapper}`);
  for (const use of ['requestAnimationFrame(t => draw(t))', 'win.requestAnimationFrame === native', 'const raf = requestAnimationFrame(cb);'])
    assert.ok(!replacing.some(rx => rx.test(use)), `and not ${use}`);
});

// The page on a virtual 60 Hz display (fake-loop.mjs): a stage and a UMAP that draw on the next animation frame when
// woken, like stage.js and umap.js, and telemetry's frame watch with a MessageChannel whose message is a task.
function page() {
  const loop = fakeLoop(), counts = { stage: 0, umap: 0, timeline: 0, batches: 0, snapshots: 0 }, listeners = {};
  const surface = key => { let raf = null; return () => { if (raf == null) raf = requestAnimationFrame(() => { raf = null; counts[key]++; }); }; };
  const stage = surface('stage'), umap = surface('umap');
  const win = {
    requestAnimationFrame: cb => globalThis.requestAnimationFrame(cb), cancelAnimationFrame: id => globalThis.cancelAnimationFrame(id),
    MessageChannel: class {
      constructor() { const port1 = { close() { this.onmessage = null; } }; this.port1 = port1; this.port2 = { postMessage: () => setTimeout(() => port1.onmessage?.({}), 0) }; }
    },
    addEventListener(type, fn) { listeners[type] = fn; }, removeEventListener() {}, setInterval: () => 1, clearInterval() {},
    document: { hidden: false },
  };
  const t = new Telemetry({ context: () => ({ mode: 'tiled' }), counters: () => ({ ...counts }), send() {}, now: () => loop.now }).start(win);
  // An input event: telemetry's capture listener runs before the page's own handler.
  const input = (type, handler) => { listeners[type](); handler?.(); };
  return { loop, counts, t, stage, umap, input, report: () => loop.with(() => t.report(win)) };
}

test('a frame watch counts the frames that showed a draw, once each however many surfaces drew, and stops when the page is still', () => {
  const p = page();
  // A one-second drag over the UMAP (one move between each two frames): the stage draws in every frame, the linked
  // UMAP highlight in the first half too.
  for (let k = 0; k < 60; k++) p.loop.at(1008 + k * 1000 / 60, () => p.input('pointermove', () => { p.stage(); if (k < 30) p.umap(); }));
  p.loop.run(2005);
  assert.equal(p.counts.stage, 60); assert.equal(p.counts.umap, 30);
  assert.equal(p.t.raf.frames, 60, 'the union of the surfaces: 60 frames, not 90 draws');
  assert.equal(p.t.watching, true);
  p.loop.run(2005 + (IDLE_FRAMES + 1) * 1000 / 60);
  assert.equal(p.t.watching, false); assert.ok(p.loop.idle(), `${IDLE_FRAMES} frames without a draw end the watch: nothing scheduled`);
  const s = p.report(), seconds = p.loop.now / 1000;
  assert.equal(s.rates.frames, Math.round(60 / seconds * 100) / 100);
  // Requested after a frame, called in the next: one frame interval at most while nothing blocks the page.
  assert.ok(s.raf_lag_ms.max <= 16.7 && s.raf_lag_ms.p95 > 8, JSON.stringify(s.raf_lag_ms));
  // A still page runs no frame callback, whatever the timer does.
  p.loop.run(30000);
  assert.ok(p.loop.idle()); assert.equal(p.report().rates.frames, 0);
});

test('a snapshot\'s panels written in a task count in the frame its stage draw shows them, not in two', () => {
  const p = page();
  // The user hovers (a watch runs); meanwhile snapshots arrive: show() writes the panels in its task and wakes the stage.
  for (let k = 0; k < 30; k++) p.loop.at(1008 + k * 1000 / 60, () => p.input('pointermove'));
  for (const at of [1105, 1205, 1305]) p.loop.at(at, () => { p.counts.snapshots++; p.counts.batches++; p.counts.timeline++; p.stage(); });
  p.loop.run(3000);
  assert.equal(p.counts.stage, 3);
  assert.equal(p.t.raf.frames, 3, 'checked after each frame: the stage drew later in the same frame');
  // Outside a watch, the busiest surface: 7 runs-list batches alone are 7 frames; a snapshot is one frame.
  p.loop.at(5000, () => { p.counts.batches += 7; });
  p.loop.at(6000, () => { p.counts.snapshots++; p.counts.batches++; p.counts.timeline++; p.stage(); });
  p.loop.run(7000);
  assert.equal(p.report().rates.frames, Math.round((3 + 7 + 1) / 7 * 100) / 100);
  // The first watched frame takes in what was drawn before the input, counted as outside a watch (the busiest surface:
  // two panel batches and this frame's stage draw are two frames), then the watch counts its own frames.
  p.loop.at(8000, () => { p.counts.batches += 2; });
  p.loop.at(9008, () => p.input('wheel', p.stage));
  p.loop.at(9058, () => p.input('wheel', p.stage));
  p.loop.run(9500);
  assert.equal(p.counts.stage, 6); assert.equal(p.t.raf.frames, 3);
});

test('long animation frames: count, longest, most blocking and the longest script, named without origins or run names; late reports are stretched', () => {
  const h = telemetry(), base = 'http://127.0.0.1:8777';
  h.advance(10000);
  h.t.longFrame(120, 70, [{ duration: 30, invoker: `IMG[src=${base}/api/v1/runs/20261016-sample-1/image/c0.png].onload`, invokerType: 'event-listener', sourceURL: `${base}/src/main.js?v=2#x`, sourceFunctionName: 'loadRunImage', sourceCharPosition: 4242 }]);
  // The production freeze: a Prior-shapes rebuild of 41k polygons, inside the stage's own animation-frame callback (an
  // anonymous arrow in stage.js: its position tells which), no longer behind a telemetry.mjs wrapper.
  h.t.longFrame(25614.34, 25564.3, [{ duration: 12, invoker: 'TimerHandler:setTimeout', invokerType: 'user-callback', sourceURL: `${base}/src/tiled-loader.mjs`, sourceFunctionName: '', sourceCharPosition: 5 },
    { duration: 25598.66, invoker: 'FrameRequestCallback', invokerType: 'user-callback', sourceURL: `${base}/src/stage.js`, sourceFunctionName: '', sourceCharPosition: 10871 }]);
  h.t.longFrame(60, 0, []);
  h.t.tick(h.win);
  const s = h.sent[0];
  assert.deepEqual(s.loaf, { count: 3, max_ms: 25614.3, blocking_max_ms: 25564.3,
    script: { ms: 25598.7, type: 'user-callback', invoker: 'FrameRequestCallback', source: '/src/stage.js', fn: '', pos: 10871 } });
  assert.equal(s.stretched, false); assert.equal(s.period_s, 10);
  // The timer was due during a 19 s task: the report came 29.16 s after the last one (production, 19:38:34).
  h.advance(29160); h.t.tick(h.win);
  assert.equal(h.sent[1].stretched, true); assert.deepEqual(h.sent[1].loaf, { count: 0, max_ms: 0, blocking_max_ms: 0, script: null });
  h.advance(PERIOD_MS * STRETCH - 1); h.t.tick(h.win); assert.equal(h.sent[2].stretched, false, 'timer jitter is not a stretch');
  assert.doesNotMatch(JSON.stringify(h.sent), /sample-1|runs|127\.0\.0\.1|cell_id|zarr|sample/i);
  // A named callback is named; a position the browser does not know (-1) is null.
  h.t.longFrame(300, 250, [{ duration: 300, invoker: 'FrameRequestCallback', invokerType: 'user-callback', sourceURL: `${base}/src/umap.js`, sourceFunctionName: 'draw', sourceCharPosition: -1 }]);
  assert.deepEqual(h.t.loaf.script, { ms: 300, type: 'user-callback', invoker: 'FrameRequestCallback', source: '/src/umap.js', fn: 'draw', pos: null });
  // Names are kept short and free of origins and attribute values, whatever invoked the script.
  assert.equal(invokerName(`IMG[src=${base}/api/v1/runs/R/image/c0.png].onload`), 'IMG[].onload');
  assert.equal(invokerName(`${base}/src/main.js`), '/src/main.js');
  assert.equal(invokerName('x'.repeat(200)).length, 80);
  assert.deepEqual([scriptPath(`${base}/src/umap.js?x=1`), scriptPath('blob:http://127.0.0.1:8777/0b1c'), scriptPath(''), scriptPath(undefined)], ['/src/umap.js', 'blob:', '', '']);
});

test('plv measures: count, longest, total and the last one with its detail; numbers, flags and words only; other measures ignored', () => {
  const h = telemetry();
  h.advance(10000);
  // As prior-layers.js and vectors-decode.mjs measure them; a detail with ids, names, lists or NaN loses those fields.
  h.t.measure('plv:prior-rebuild', 25910.24, { polygons: 41190, layer: 'xenium' });
  h.t.measure('plv:prior-rebuild', 120.5, { polygons: 82380, layer: 'xenium+nuclei', run: '20261016-203634-sample-1', sample: 'sample-1', box: [0, 0, 1, 1], ok: true, at: NaN });
  h.t.measure('plv:vectors-decode', 830, { bytes: 9876543, polygons: 82380 });
  h.t.measure('plv:tile-upload', 4, 75.6);
  h.t.measure('next-route', 4);                         // not the app's ('plv:' only)
  h.t.measure('plv:', 1); h.t.measure('plv:a b/c', 2);   // malformed names are summed as plv:other
  h.t.tick(h.win);
  assert.deepEqual(h.sent[0].measures, {
    'plv:prior-rebuild': { n: 2, max_ms: 25910.2, total_ms: 26030.7, last_ms: 120.5, last: { polygons: 82380, layer: 'xenium+nuclei', ok: true } },
    'plv:vectors-decode': { n: 1, max_ms: 830, total_ms: 830, last_ms: 830, last: { bytes: 9876543, polygons: 82380 } },
    'plv:tile-upload': { n: 1, max_ms: 4, total_ms: 4, last_ms: 4, last: 75.6 },
    'plv:other': { n: 2, max_ms: 2, total_ms: 3, last_ms: 2, last: null },
  });
  h.advance(10000); h.t.tick(h.win); assert.deepEqual(h.sent[1].measures, {}, 'a new period starts empty');
  assert.deepEqual([measureName('plv:prior-rebuild'), measureName('plv:x'.padEnd(60, 'x')), measureName('plv:9'), measureName('plv:sample-1 run')], ['plv:prior-rebuild', 'plv:other', 'plv:other', 'plv:other']);
  assert.deepEqual([measureDetail(null), measureDetail(undefined), measureDetail('cells'), measureDetail('sample-1'), measureDetail([1, 2]), measureDetail(Infinity), measureDetail(false)],
    [null, null, 'cells', null, null, null, false]);
  assert.deepEqual(measureDetail({ a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7 }), { a: 1, b: 2, c: 3, d: 4 });
  assert.deepEqual(measureDetail({ 'bad key': 1, ['k'.repeat(30)]: 2, layer: 'x'.repeat(25), nested: { a: 1 } }), {});
  // At most 8 names a report (then plv:other), so a page that measures many steps still sends a small report.
  for (let k = 0; k < 12; k++) h.t.measure(`plv:step-${'abcdefghijkl'[k]}`, k);
  assert.equal(Object.keys(h.t.measures).length, 9); assert.equal(h.t.measures['plv:other'].n, 4);
});

test('start() observes long animation frames and measures where the browser has them, and reports null for them where it does not', () => {
  const observers = [], timers = [];
  const PerformanceObserver = class {
    static supportedEntryTypes = ['longtask', 'event', 'resource', 'long-animation-frame', 'measure'];
    constructor(fn) { this.fn = fn; observers.push(this); }
    observe(options) { this.options = options; } disconnect() { this.disconnected = true; }
  };
  const win = { PerformanceObserver, setInterval(fn) { timers.push(fn); return timers.length; }, clearInterval() {}, document: { hidden: false } };
  let clock = 0;
  const sent = [], t = new Telemetry({ context: () => ({ mode: 'tiled' }), send: body => sent.push(JSON.parse(body)), now: () => clock }).start(win);
  const loaf = observers.find(o => o.options.type === 'long-animation-frame');
  assert.ok(loaf); assert.equal(loaf.options.buffered, true);
  loaf.fn({ getEntries: () => [{ duration: 25614, blockingDuration: 25564, scripts: [{ duration: 25600, invoker: 'FrameRequestCallback', invokerType: 'user-callback', sourceURL: 'http://127.0.0.1:8777/src/stage.js', sourceFunctionName: '', sourceCharPosition: 10871 }] }] });
  clock = 10000; timers[0]();
  assert.deepEqual(sent[0].loaf.script, { ms: 25600, type: 'user-callback', invoker: 'FrameRequestCallback', source: '/src/stage.js', fn: '', pos: 10871 });
  assert.deepEqual(sent[0].measures, {});
  t.stop(); assert.ok(loaf.disconnected);
  // Chrome before 123 (no long-animation-frame entry type), or no PerformanceObserver at all: the fields are null.
  for (const old of [{ ...win, PerformanceObserver: class extends PerformanceObserver { static supportedEntryTypes = ['longtask'] } }, { ...win, PerformanceObserver: undefined }]) {
    observers.length = timers.length = 0; sent.length = 0;
    const o = new Telemetry({ context: () => ({ mode: 'detail' }), send: body => sent.push(JSON.parse(body)), now: () => clock }).start(old);
    assert.equal(observers.some(x => ['long-animation-frame', 'measure'].includes(x.options?.type)), false);
    clock += 10000; timers[0](); assert.equal(sent[0].loaf, null); assert.equal(sent[0].measures, null); o.stop();
  }
});

test('a report with the longest names and the most measures still fits the engine limit', () => {
  const h = telemetry(), long = 'x'.repeat(500);
  h.advance(10000);
  h.t.longFrame(1e7, 1e7, [{ duration: 1e7, invoker: long, invokerType: long, sourceURL: `http://127.0.0.1:8777/${long}`, sourceFunctionName: long, sourceCharPosition: 2 ** 31 }]);
  for (const kind of ['view', 'frame', 'image', 'transcripts', 'vectors', 'masks', 'shapes']) h.t.fetched(kind, 1e9, 1e10);
  const detail =Object.fromEntries(Array.from({ length: 20 }, (_, i) => [`field_${'abcdefghijklmnopqrst'[i]}`.padEnd(24, 'z'), 'w'.repeat(24)]));
  for (let k = 0; k < 20; k++) h.t.measure(`plv:${'s'.repeat(30)}${'abcdefghijklmnopqrst'[k]}`, 1e7 + k / 3, detail);
  h.t.tick(h.win);
  const bytes = new TextEncoder().encode(JSON.stringify(h.sent[0])).length;
  assert.ok(bytes < 6 * 1024, `${bytes} bytes: well inside the engine's ${MAX_BYTES}`);
  assert.ok([h.sent[0].loaf.script.invoker, h.sent[0].loaf.script.source, h.sent[0].loaf.script.fn].every(text => text.length <= 80));
  assert.equal(Object.keys(h.sent[0].measures).length, 9);
});

test('the default transport is a keepalive JSON POST whose failures are swallowed', async () => {
  const calls = [], fetch = globalThis.fetch;
  try {
    globalThis.fetch = (url, options) => { calls.push([url, options]); return Promise.reject(new Error('engine closed')); };
    assert.doesNotThrow(() => post('{"v":1}'));
    globalThis.fetch = () => { throw new Error('no fetch'); };
    assert.doesNotThrow(() => post('{"v":1}'));
    await new Promise(resolve => setImmediate(resolve));   // the rejected promise is handled
  } finally { globalThis.fetch = fetch; }
  assert.equal(calls.length, 1);
  const [url, options] = calls[0];
  assert.equal(url, ENDPOINT); assert.equal(options.method, 'POST'); assert.equal(options.keepalive, true);
  assert.equal(options.headers['Content-Type'], 'application/json'); assert.equal(options.body, '{"v":1}');
  // The page's fetch adds the page's key (page-key.mjs), and the request stays in the default cors mode, which sends the
  // page's real Origin (a no-cors request under the app's no-referrer policy would carry Origin: null, which the engine
  // refuses).
  assert.equal(options.credentials, 'same-origin'); assert.equal(options.mode, undefined);
});

test('the page and the engine agree on the endpoint and the size limit; a report over the limit is not sent', () => {
  // server/app.py: base = "/api/v1", @app.post(base + "/telemetry"), TELEMETRY_MAX_BYTES = 16 * 1024 (413 beyond).
  const app = readFileSync(new URL('../../server/app.py', import.meta.url), 'utf8');
  const base = app.match(/^\s*base = "([^"]+)"/m)?.[1], route = app.match(/@app\.post\(base \+ "(\/telemetry)"\)/)?.[1];
  assert.equal(base + route, ENDPOINT);
  const limit = app.match(/^TELEMETRY_MAX_BYTES = ([\d\s*]+?)\s*(?:#|$)/m)?.[1];
  assert.ok(limit, 'TELEMETRY_MAX_BYTES is defined');
  assert.equal(limit.split('*').reduce((product, factor) => product * Number(factor), 1), MAX_BYTES);
  // Measured in UTF-8 bytes, as the engine counts them: a renderer name of 2-byte characters near the limit.
  const sizes = renderer => {
    const h = telemetry(() => ({ mode: 'detail', state: 'running', cells: 10, live: true, umap: false, view: '2d', level: null, stage: [10, 10], renderer }));
    h.advance(10000); h.t.tick(h.win); return h.sent;
  };
  assert.equal(sizes('ANGLE (NVIDIA Example GPU)').length, 1);
  assert.equal(sizes('é'.repeat(MAX_BYTES / 2)).length, 0, 'over the limit in bytes (not in characters): not sent');
  const fits = sizes('é'.repeat(MAX_BYTES / 2 - 600));
  assert.equal(fits.length, 1);
  assert.ok(new TextEncoder().encode(JSON.stringify(fits[0])).length <= MAX_BYTES);
});
