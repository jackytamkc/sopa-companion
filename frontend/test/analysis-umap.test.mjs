import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { domFixture } from './dom-fixture.mjs';
// The real analysis controller driving the real UMAP view and legend, fed summaries shaped like the GPU worker's
// (analysis/worker.py: summary.bounds and summary.clusters[].centroid), in a minimal fake browser.
const url = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const read = name => readFile(new URL(`../src/${name}`, import.meta.url), 'utf8');
const link = (source, imports) => Object.entries(imports).reduce((s, [from, to]) => {
  assert.ok(s.includes(`'${from}'`), `import ${from}`); return s.replace(`'${from}'`, JSON.stringify(to));
}, source);
const clustersURL = url(await read('clusters.js')), helpersURL = url(await read('analysis-helpers.mjs'));
const hookURL = new URL('../src/harness-hook.mjs', import.meta.url).href, logURL = new URL('../src/log-export.mjs', import.meta.url).href;
const umapURL = url(link(await read('umap.js'), { './clusters.js': clustersURL, './harness-hook.mjs': hookURL }));
// The Advanced settings' module by its file URL, so its own import of explain.mjs resolves beside it.
const settingsURL = new URL('../src/analysis-settings.mjs', import.meta.url).href;
// The deck and the statistics' helpers by their file URLs too (the badge's words, a cell's count in proseg's count).
const copyURL = new URL('../src/qc-copy.mjs', import.meta.url).href, statsURL = new URL('../src/cellstats.mjs', import.meta.url).href;
const { Analysis } = await import(url(link(await read('analysis.js'), { './analysis-helpers.mjs': helpersURL, './umap.js': umapURL, './clusters.js': clustersURL,
  './harness-hook.mjs': hookURL, './log-export.mjs': logURL, './analysis-settings.mjs': settingsURL, './qc-copy.mjs': copyURL, './cellstats.mjs': statsURL })));
const { fitBounds } = await import(umapURL);

function browser() {
  const { document, Element } = domFixture(), env = { clock: 1000, frames: new Map(), next: 1 };
  Object.assign(Element.prototype, {
    remove() { const p = this.parentElement; if (p) p.childNodes = p.childNodes.filter(n => n !== this); this.parentElement = null; },
    removeEventListener(type, fn) { this.listeners.set(type, (this.listeners.get(type) || []).filter(f => f !== fn)); },
    setPointerCapture() {}, releasePointerCapture() {},
  });
  const gl = new Proxy({}, { get: (_, name) => typeof name !== 'string' ? undefined : name === name.toUpperCase() ? 1
    : (...args) => /Parameter$/.test(name) ? true : /^create|^getUniformLocation$/.test(name) ? { name, args } : undefined });
  globalThis.document = document;
  globalThis.window = Object.assign(new Element('window'), { devicePixelRatio: 1 });
  globalThis.requestAnimationFrame = fn => { const id = env.next++; env.frames.set(id, fn); return id; };
  globalThis.cancelAnimationFrame = id => env.frames.delete(id);
  globalThis.performance = { now: () => env.clock };
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  env.pane = document.createElement('section');
  env.canvas = Object.assign(document.createElement('canvas'), { width: 300, height: 150, offsetLeft: 0, offsetTop: 28,
    getContext: kind => kind === 'webgl2' ? gl : null, getBoundingClientRect: () => ({ left: 10, top: 40, width: 400, height: 300 }) });
  env.pane.append(env.canvas);
  env.frame = (ms = 16) => { env.clock += ms; const due = [...env.frames.values()]; env.frames.clear(); due.forEach(fn => fn(env.clock)); };
  env.settle = () => { let n = 0; while (env.frames.size && n < 1000) { env.frame(); n++; } return n; };
  const state = { seq: 5, live: true }, calls = { setAnalysis: [], setHidden: [] }, a = Object.create(Analysis.prototype);
  Object.assign(a, { run: 'r', epoch: 1, enabled: true, results: new Map(), available: [], hidden: new Set(), appliedAt: -Infinity,
    hoverLabel: 0, hoverCluster: null, selectedLabel: 0, canvas: env.canvas, panel: { render() {} }, select() {},
    legend: document.createElement('div'), tip: document.createElement('div'), controls: document.createElement('div'),
    empty: document.createElement('p'), supported: true, coloured: true, shownNote: '',
    context: () => ({ seq: state.seq, live: state.live, frames: [], cells: [] }),
    stage: { options: {}, set(o) { Object.assign(this.options, o); }, setAnalysis: r => calls.setAnalysis.push(r?.seq ?? null),
      clusterGPU: { setHidden: hidden => calls.setHidden.push([...hidden]) } } });
  return { env, a, state, calls };
}

/** A result as the controller holds it: the /cells payload decoded, plus the worker's summary for it. Cluster c sits
 * around (6 * (c % 4) - 9, 6 * floor(c / 4) - 3); every 101st cell is not analysed (cluster -1, NaN coordinates). */
function workerResult(seq, n, { k = 8, bounds = true, centroids = true } = {}) {
  const clusters = new Int16Array(n), coords = new Float32Array(3 * n), sizes = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    const c = i % 101 === 0 ? -1 : i % k;
    clusters[i] = c;
    if (c < 0) { coords.fill(NaN, 3 * i, 3 * i + 3); continue; }
    sizes[c]++;
    coords[3 * i] = 6 * (c % 4) - 9 + (i % 13) / 6.5 - 1; coords[3 * i + 1] = 6 * Math.floor(c / 4) - 3 + (i % 7) / 3.5 - 1;
  }
  const summary = { seq, iter: 10 * seq, ncells: n - Math.ceil(n / 101), nclusters: k, umap_dims: 2,
    clusters: sizes.map((size, c) => ({ id: c, size, color: c, label: `G${c}a · G${c}b`, top: [],
      ...(centroids ? { centroid: c === k - 1 ? null : [6 * (c % 4) - 9, 6 * Math.floor(c / 4) - 3, 0] } : {}) })) };
  if (bounds) summary.bounds = { x: [-9.75, 8.75], y: [-3.5, 3.5] };
  return { clusters, coords, seq, summary };
}
function sortLengths(fn) {
  const lengths = [], TA = Object.getPrototypeOf(Int8Array.prototype), typed = TA.sort, array = Array.prototype.sort;
  TA.sort = function (...args) { lengths.push(this.length); return typed.apply(this, args); };
  Array.prototype.sort = function (...args) { lengths.push(this.length); return array.apply(this, args); };
  try { fn(); return lengths; } finally { TA.sort = typed; Array.prototype.sort = array; }
}
const rows = a => a.legend.childNodes[0].childNodes[1].childNodes;   // the legend: its heading, then its rows

test('a worker summary drives the real view: its bounds fit the map, its centroids anchor the labels, nothing is sorted', () => {
  const { env, a, calls } = browser(), r = workerResult(5, 40000);
  const lengths = sortLengths(() => { a.results.set(5, r); a.frame(); env.frame(); });
  assert.deepEqual(lengths, []);                                         // no per-point sort anywhere for 40k cells
  const view = a.view;
  assert.equal(a.applied, r); assert.deepEqual(calls.setAnalysis, [5]);
  assert.deepEqual(view.bounds, r.summary.bounds);
  assert.deepEqual(view.center, [-0.5, 0, 0]);                           // fitted to the worker's view box
  for (const c of r.summary.clusters) assert.deepEqual(view.centroids.get(c.id), c.centroid ?? undefined);
  // A cluster whose centroid is null has no analysed point to anchor a label (contract), so it gets no label.
  assert.deepEqual(view.labels.map(l => [l.id, l.text]), r.summary.clusters.slice(0, -1).map(c => [c.id, c.label]));
  assert.equal(rows(a).length, 8);
  assert.deepEqual(rows(a).map(row => row.dataset.cluster), ['0', '1', '2', '3', '4', '5', '6', '7']);
  rows(a)[2].click();                                                    // hide cluster 2: UMAP and tissue both follow
  assert.equal(view.hiddenClusters, a.hidden); assert.deepEqual([...a.hidden], [2]); assert.deepEqual(calls.setHidden.at(-1), [2]);
  assert.equal(view.weights[2], Math.fround(.12)); assert.equal(view.weights[3], 1);
  a.tissueHover(4);                                                      // label 4 = cell 3, in cluster 3
  assert.equal(view.clusterHighlight, 3); assert.equal(view.weights[3], 1); assert.equal(view.weights[4], Math.fround(.12));
  a.tissueHover(0); assert.equal(view.clusterHighlight, null);
  env.settle(); a.frame(); a.frame();                                    // main.js calls frame() several times per frame
  assert.equal(env.frames.size, 0); assert.deepEqual(calls.setAnalysis, [5]);   // nothing new: the view stays idle
});

test('a summary saved by an older worker (no bounds, no centroids) still fits and labels the view itself', () => {
  const { env, a } = browser(), r = workerResult(5, 3000, { bounds: false, centroids: false });
  a.results.set(5, r); a.frame(); env.frame();
  const view = a.view, b = fitBounds(view.to, view.count);
  assert.equal(view.bounds, null);
  assert.deepEqual(view.center, [(b.x[0] + b.x[1]) / 2, (b.y[0] + b.y[1]) / 2, 0]);
  assert.equal(view.labels.length, 8);
  for (const [id, [x, y, z]] of view.centroids) {                       // sampled medians near each cluster's centre
    assert.ok(Math.abs(x - (6 * (id % 4) - 9)) < .3 && Math.abs(y - (6 * Math.floor(id / 4) - 3)) < .3 && z === 0, `${id}: ${x}, ${y}`);
  }
});

test('a live whole-sample stream reaches the real view at most once a second; seeking back before it shows the oldest, and none left clears it', () => {
  const { env, a, state, calls } = browser();
  state.seq = 1; a.results.set(1, workerResult(1, 40000)); a.frame();
  const view = a.view, t0 = env.clock;
  assert.ok(view.duration > 0);                                          // the first layout flies in
  for (let k = 0; k < 3; k++) env.frame();
  env.clock = t0 + 300; state.seq = 2; a.results.set(2, workerResult(2, 40000)); a.frame();
  assert.equal(a.applied.seq, 1); assert.notEqual(a.applyTimer, null);   // held back: not handed to the view
  assert.deepEqual(calls.setAnalysis, [1]);
  const flying = view.started; env.frame(); assert.equal(view.started, flying);   // the flight in progress goes on
  env.clock = t0 + 1000; a.frame();                                      // the window has passed: the newest applies
  assert.equal(a.applied.seq, 2); assert.equal(a.applyTimer, null); assert.deepEqual(calls.setAnalysis, [1, 2]);
  assert.ok(view.duration > 0); assert.ok(env.settle() > 40);          // 40k cells fly to the new layout too
  // Scrubbed before the first result: the oldest after it (as the engine answers a frame asked for: the nearest with
  // proseg's count); with none left (another window's settings dropped them) the view clears.
  state.live = false; state.seq = 0; a.frame(); a.stopTimers();
  assert.equal(a.applied.seq, 1); assert.deepEqual(calls.setAnalysis, [1, 2, 1]); env.settle();
  a.results.clear(); a.frame(); a.stopTimers();
  assert.equal(a.applied, null); assert.deepEqual(calls.setAnalysis, [1, 2, 1, null]);
  assert.equal(view.labels.length, 0); assert.equal(a.legend.childNodes.length, 0);
  assert.ok(Array.from({ length: view.count }, (_, i) => view.to[4 * i + 3]).every(w => w === 0));   // every point fades out
  env.settle(); assert.equal(env.frames.size, 0);
});

/** The /cells payload of a result (decodeAnalysis): its clusters (i16), then its coordinates (f32 x, y, z), little-endian. */
function cellsPayload(r) {
  const n = r.clusters.length, buffer = new ArrayBuffer(n * 14), view = new DataView(buffer);
  r.clusters.forEach((c, i) => view.setInt16(i * 2, c, true));
  r.coords.forEach((v, i) => view.setFloat32(n * 2 + i * 4, v, true));
  return buffer;
}

test('under Compare (setCovered) the pane holds still: a flight on its way stops; a result pushed meanwhile asks no frame; uncovered, the newest flies in once', async () => {
  const { env, a, state, calls } = browser(), fetched = [], saved = globalThis.fetch;
  globalThis.fetch = async url => { fetched.push(url); return { ok: true, arrayBuffer: async () => cellsPayload(workerResult(3, 3000, { k: 6 })) }; };
  try {
    state.live = false; state.seq = 1; a.results.set(1, workerResult(1, 3000)); a.frame(); env.settle();
    const view = a.view;
    state.seq = 2; a.results.set(2, workerResult(2, 3000, { k: 5 })); a.frame();
    assert.equal(env.frames.size, 1, 'a new layout flies');
    a.setCovered(true);
    assert.equal(env.frames.size, 0, 'its frame goes: nothing drawn under Compare');
    // The frame on screen's analysis lands meanwhile (the engine's push, its cells downloaded): kept, not applied.
    state.seq = 3; env.clock += 2000; const draws = view.drawCount, version = view.version;
    a.event({ run: 'r', state: 'ready', seq: 3, iter: 30, summary: workerResult(3, 3000, { k: 6 }).summary });
    for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(fetched, ['/api/v1/runs/r/analysis/3/cells']); assert.ok(a.results.has(3), 'kept');
    assert.equal(a.applied.seq, 2); assert.deepEqual(calls.setAnalysis, [1, 2], 'the tissue keeps its colours');
    assert.equal(view.version, version, 'no new flight'); assert.equal(env.frames.size, 0, 'no animation frame asked for');
    view.resize(); view.ringCells([[5, 1]]); a.frame();                  // a window resize, Go deeper's rings, a frame shown
    assert.equal(env.frames.size, 0); assert.equal(view.drawCount, draws);
    // Uncovered: the newest for the frame on screen is applied once, and flies in.
    a.setCovered(false);
    assert.equal(a.applied.seq, 3); assert.deepEqual(calls.setAnalysis, [1, 2, 3]); assert.equal(view.version, version + 1);
    assert.ok(env.settle() > 0); assert.ok(view.drawCount > draws);
    a.setCovered(false); a.frame(); assert.deepEqual(calls.setAnalysis, [1, 2, 3], 'once'); assert.equal(env.frames.size, 0);
    // Covered as the pane is made (a run's first result under Compare): its first frame waits too. Uncovered without a
    // catch-up (another run or the picker opens instead), nothing held is applied.
    const b = browser();
    b.a.setCovered(true); b.a.ensureView();
    assert.equal(b.env.frames.size, 0);
    b.state.live = false; b.a.results.set(5, workerResult(5, 300)); b.a.frame();
    assert.equal(b.a.applied, undefined); assert.equal(b.env.frames.size, 0);
    b.a.setCovered(false, false);
    assert.equal(b.a.applied, undefined); assert.ok(b.env.frames.size <= 1, 'at most what the view had pending');
    b.env.settle(); b.a.frame(); assert.equal(b.a.applied.seq, 5, 'the next frame shows it');
    a.stopTimers(); b.a.stopTimers();
  } finally { globalThis.fetch = saved; }
});
