import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
// UmapView end to end in a minimal browser: DOM nodes that count layout reads and style writes, a WebGL2 context that
// records its calls, and animation frames that run only when the test steps them.
const url = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const paletteURL = url(await readFile(new URL('../src/clusters.js', import.meta.url), 'utf8'));
const source = (await readFile(new URL('../src/umap.js', import.meta.url), 'utf8')).replace("'./clusters.js'", JSON.stringify(paletteURL))
  .replace("'./harness-hook.mjs'", JSON.stringify(new URL('../src/harness-hook.mjs', import.meta.url).href));
const { UmapView, FLIGHT_MS, FLIGHT_MIN_MS, FLIGHT_FILL } = await import(url(source));

const RECT = { left: 10, top: 40, width: 400, height: 300 };
function browser() {
  const env = { clock: 1000, frames: new Map(), next: 1, layoutReads: 0, rectReads: 0, styleWrites: 0, created: 0, gl: [], observers: [] };
  const style = () => new Proxy({}, { set(o, k, v) { env.styleWrites++; o[k] = v; return true; } });
  class Node {
    constructor(tag) { this.tagName = tag.toUpperCase(); this.childNodes = []; this.parentElement = null; this.listeners = new Map(); this.style = style(); env.created++; }
    append(...nodes) { for (const node of nodes) { node.remove(); node.parentElement = this; this.childNodes.push(node); } }
    remove() { const p = this.parentElement; if (p) { p.childNodes = p.childNodes.filter(n => n !== this); this.parentElement = null; } }
    addEventListener(type, fn) { this.listeners.set(type, [...(this.listeners.get(type) || []), fn]); }
    removeEventListener(type, fn) { this.listeners.set(type, (this.listeners.get(type) || []).filter(f => f !== fn)); }
    emit(type, props = {}) { const event = { preventDefault() {}, ...props }; for (const fn of this.listeners.get(type) || []) fn(event); return event; }
    get offsetWidth() { env.layoutReads++; return 80; }
    get offsetHeight() { env.layoutReads++; return 20; }
    get offsetLeft() { env.layoutReads++; return 0; }
    get offsetTop() { env.layoutReads++; return 30; }
    getBoundingClientRect() { env.layoutReads++; return { left: 0, top: 0, width: 0, height: 0 }; }
  }
  const gl = new Proxy({ VERTEX_SHADER: 1, FRAGMENT_SHADER: 2, COMPILE_STATUS: 3, LINK_STATUS: 4, ARRAY_BUFFER: 5, DYNAMIC_DRAW: 6, FLOAT: 7, COLOR_BUFFER_BIT: 8, DEPTH_BUFFER_BIT: 16, DEPTH_TEST: 9, BLEND: 10, SRC_ALPHA: 11, ONE_MINUS_SRC_ALPHA: 12, POINTS: 0 }, {
    get: (constants, name) => name in constants ? constants[name] : (...args) => {
      env.gl.push([name, ...args]);
      return /Parameter$/.test(name) ? true : /^create|^getUniformLocation$/.test(name) ? { name } : undefined;
    },
  });
  class Canvas extends Node {
    constructor() { super('canvas'); this.width = 300; this.height = 150; this.rect = RECT; }
    getContext(kind) { return kind === 'webgl2' ? gl : null; }
    getBoundingClientRect() { env.layoutReads++; env.rectReads++; return { ...this.rect }; }
    setPointerCapture() {} releasePointerCapture() {}
  }
  globalThis.document = { body: new Node('body'), createElement: tag => new Node(tag), createTextNode: text => Object.assign(new Node('#text'), { textContent: text }) };
  globalThis.window = Object.assign(new Node('window'), { devicePixelRatio: 1 });
  globalThis.requestAnimationFrame = fn => { const id = env.next++; env.frames.set(id, fn); return id; };
  globalThis.cancelAnimationFrame = id => env.frames.delete(id);
  globalThis.performance = { now: () => env.clock };
  globalThis.ResizeObserver = class { constructor(fn) { this.fn = fn; env.observers.push(this); } observe(node) { this.node = node; } disconnect() { this.disconnected = true; } };
  env.pane = new Node('section'); document.body.append(env.pane);
  env.canvas = new Canvas(); env.pane.append(env.canvas);
  env.frame = (ms = 16) => { env.clock += ms; const due = [...env.frames.values()]; env.frames.clear(); due.forEach(fn => fn(env.clock)); return due.length; };
  env.settle = () => { let n = 0; while (env.frames.size && n < 1000) { env.frame(); n++; } return n; };
  env.calls = name => env.gl.filter(call => call[0] === name).length;
  return env;
}
const rng = seed => () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
function data(n, { seed = 1, shift = 0 } = {}) {
  const rand = rng(seed), clusters = new Int16Array(n), coords = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const c = i % 6; clusters[i] = i % 97 ? c : -1;
    coords[3 * i] = c * 3 + rand() * 2 + shift; coords[3 * i + 1] = (c % 3) * 2 + rand() * 2; coords[3 * i + 2] = 0;
  }
  return { clusters, coords };
}
/** Screen position (canvas CSS px) of point i's target, and the previous every-point hover scan at that pointer. */
function screen(view, i) {
  const x = view.to[4 * i] - view.center[0], y = view.to[4 * i + 1] - view.center[1], scale = view.height * 1.20710678 / view.distance;
  return { x: view.width / 2 + x * scale, y: view.height / 2 - y * scale };
}
function scanPick(view, x, y) {
  let best = null, bestScore = Infinity;
  for (let i = 0; i < view.count; i++) {
    if (view.to[4 * i + 3] < .15 || view.hiddenClusters?.has(view.clusters[i])) continue;
    const p = screen(view, i), d = Math.hypot(p.x - x, p.y - y);
    if (p.x < 0 || p.y < 0 || p.x > view.width || p.y > view.height) continue;
    if (d <= 5 && d < bestScore - .5) { best = i; bestScore = d; }
  }
  return best;
}
function sortLengths(fn) {
  const lengths = [], TA = Object.getPrototypeOf(Int8Array.prototype), typed = TA.sort, array = Array.prototype.sort;
  TA.sort = function (...args) { lengths.push(this.length); return typed.apply(this, args); };
  Array.prototype.sort = function (...args) { lengths.push(this.length); return array.apply(this, args); };
  try { fn(); return lengths; } finally { TA.sort = typed; Array.prototype.sort = array; }
}
const labelsOf = view => view.overlay.childNodes.filter(node => node !== view.gizmo);
const move = (env, p, extra = {}) => env.canvas.emit('pointermove', { clientX: RECT.left + p.x, clientY: RECT.top + p.y, pointerId: 1, ...extra });

test('every new layout flies, at any map size, and the view is idle again once it has landed', () => {
  const env = browser(), view = new UmapView(env.canvas);
  assert.equal(env.frames.size, 1); env.frame();                       // the first frame measures the canvas
  assert.equal(env.frames.size, 0); assert.equal(env.calls('clear'), 1);
  assert.equal(view.overlay.parentElement, env.pane);                  // labels sit in the canvas's own container
  assert.deepEqual(['left', 'top', 'width', 'height'].map(k => view.overlay.style[k]), ['0px', '30px', '400px', '300px']);
  view.setData(data(25000));
  assert.equal(view.duration, FLIGHT_MS);                              // a whole-sample-sized map flies too
  const flight = env.settle();
  assert.ok(flight >= 56 && flight <= 58, `${flight} frames for a ${FLIGHT_MS} ms flight`);
  assert.equal(env.calls('clear'), 1 + flight);                        // one draw per frame of the flight
  view.setData(data(25000, { seed: 2, shift: 1 }));                    // the next layout flies from where the dots are
  assert.ok(env.settle() > 40); assert.equal(env.frames.size, 0);
  const clears = env.calls('clear'), reads = env.layoutReads;
  env.clock += 10000; env.frame();
  assert.equal(env.calls('clear'), clears); assert.equal(env.layoutReads, reads);   // nothing changed: nothing ran
  assert.equal(env.rectReads, 1);                                       // the canvas rect was read once, not per frame
  window.emit('scroll'); assert.equal(env.frames.size, 0);              // scrolling elsewhere does not redraw
});

test('a flight starts from where the dots are drawn; while layouts stream in, the dots keep moving', () => {
  const env = browser(), view = new UmapView(env.canvas);
  env.frame(); view.setData(data(10000)); env.settle();
  env.clock += 1500;                                                    // an occasional update: a step, a paused run
  view.setData(data(10000, { seed: 2, shift: 5 }));
  assert.equal(view.duration, FLIGHT_MS);
  for (let k = 0; k < 5; k++) env.frame();
  const t = view.transition(env.clock), x = view.from[20] + (view.to[20] - view.from[20]) * t, next = data(10000, { seed: 3, shift: -5 });
  assert.ok(t > 0 && t < 1);
  view.setData(next);                                                   // a new layout 80 ms into the flight:
  assert.equal(view.duration, FLIGHT_MIN_MS);                           // a short flight,
  assert.ok(Math.abs(view.from[20] - x) < 1e-4, `${view.from[20]} vs ${x}`);   // from the state on screen
  assert.equal(view.to[20], next.coords[15]);
  const landing = env.settle();
  assert.ok(landing >= 12 && landing <= 14, `${landing} frames`); assert.equal(env.frames.size, 0);
  // Back-to-back live analyses (a whole sample about once a second, a region every 0.1-0.5 s): each flight fills
  // FLIGHT_FILL of the interval (at most FLIGHT_MS), so the map is in motion nearly all the time.
  for (const every of [250, 400, 600, 1200]) {
    let moving = 0, frames = 0;
    for (let k = 0; k < 8; k++) {
      view.setData(data(10000, { seed: 10 + k, shift: k }));
      for (let ms = 0; ms < every; ms += 16, frames++) if (env.frame()) moving++;
    }
    const expect = Math.min(FLIGHT_FILL, FLIGHT_MS / every);
    assert.ok(Math.abs(moving / frames - expect) < .1, `a layout every ${every} ms: drawing ${moving} of ${frames} frames`);
    env.settle();
  }
});

test('a large map flies to its new layout; a resting hover and the selection land with it', () => {
  const env = browser(), hovers = [];
  const view = new UmapView(env.canvas, { onHover: id => hovers.push(id) });
  env.frame(); view.setData(data(40000)); env.settle();
  const p = screen(view, 4321); env.canvas.emit('pointerenter'); move(env, p); env.frame();
  assert.equal(view.hover, scanPick(view, p.x, p.y)); assert.notEqual(view.hover, null);
  view.select(77); env.settle();
  const clears = env.calls('clear'), picks = view.drawnPicks, churn = hovers.length;
  view.setData(data(40000, { seed: 9, shift: 2 }));                    // every point moves under the resting pointer
  const flight = env.settle();
  assert.ok(flight > 50, `${flight} frames`); assert.equal(env.calls('clear'), clears + flight);
  assert.equal(view.drawnPicks, picks, 'a resting pointer: no scans of moving points');
  assert.equal(view.hover, scanPick(view, p.x, p.y), 'once landed, the hover names the point under the pointer');
  assert.ok(hovers.length <= churn + 1); assert.equal(hovers.at(-1), view.hover);
  assert.equal(view.selected, 77); assert.equal(view.transition(env.clock), 1);
  const click = screen(view, 999), at = { clientX: RECT.left + click.x, clientY: RECT.top + click.y, pointerId: 1 };
  env.canvas.emit('pointerdown', { ...at, button: 0 }); env.canvas.emit('pointerup', at);
  assert.equal(view.selected, scanPick(view, click.x, click.y));
});

test('hover is picked once per frame after pointer moves, from a grid rebuilt only for new data or a new camera', async () => {
  const env = browser(), hovers = [], selects = [];
  const view = new UmapView(env.canvas, { onHover: id => hovers.push(id), onSelect: id => selects.push(id) });
  env.frame(); view.setData(data(40000)); env.settle();
  assert.equal(view.gridBuilds, 0);                                     // nobody has pointed at the map yet
  const p = screen(view, 1234), reads = env.rectReads;
  env.canvas.emit('pointerenter');
  for (const dx of [30, 20, 0]) move(env, { x: p.x + dx, y: p.y });
  assert.equal(env.frames.size, 1); assert.equal(env.rectReads, reads + 1);
  env.frame();
  assert.equal(view.gridBuilds, 1); assert.notEqual(view.hover, null);
  assert.equal(view.hover, scanPick(view, p.x, p.y)); assert.deepEqual(hovers, [view.hover]);
  const rand = rng(4);
  for (let k = 0; k < 30; k++) {
    const q = k % 2 ? screen(view, Math.floor(rand() * 40000)) : { x: rand() * 400, y: rand() * 300 };
    move(env, q); move(env, q); env.frame();
    assert.equal(view.hover, scanPick(view, q.x, q.y));
  }
  assert.equal(view.gridBuilds, 1); assert.equal(env.rectReads, reads + 1);
  const target = screen(view, 777); move(env, target); env.frame();
  view.setData(data(40000, { seed: 9, shift: .5 }));                    // a new result under a resting pointer
  assert.ok(env.settle() > 1);                                          // it flies
  assert.equal(view.gridBuilds, 2);                                     // once for the new targets, when it lands
  assert.equal(view.hover, scanPick(view, target.x, target.y));
  const click = screen(view, 4321), at = { clientX: RECT.left + click.x, clientY: RECT.top + click.y, pointerId: 1 };
  env.canvas.emit('pointerdown', { ...at, button: 0 }); env.canvas.emit('pointerup', at);
  assert.equal(selects.length, 1); assert.equal(selects[0], scanPick(view, click.x, click.y)); assert.equal(view.selected, selects[0]);
  assert.notEqual(selects[0], null);
  env.settle();
  for (let k = 0; k < 4; k++) { env.canvas.emit('wheel', { ...at, deltaY: -120, deltaMode: 0 }); env.frame(); }
  assert.equal(view.gridBuilds, 2);                                     // zooming redraws but does not re-pick each frame
  assert.equal(view.userCamera, true);
  await new Promise(resolve => setTimeout(resolve, 150)); env.settle();
  assert.equal(view.gridBuilds, 3); assert.equal(view.hover, scanPick(view, click.x, click.y));
  env.canvas.emit('pointerleave');
  assert.equal(view.hover, null); assert.equal(hovers.at(-1), null);
  env.settle(); const builds = view.gridBuilds;
  view.setData(data(40000, { seed: 10 })); env.settle();
  assert.equal(view.gridBuilds, builds);                                // no pointer: no picking at all
});

test('during a morph clicks and a moving pointer pick the point as drawn; hover settles on the target when it ends', () => {
  const env = browser(), hovers = [], selects = [];
  const view = new UmapView(env.canvas, { onHover: id => hovers.push(id), onSelect: id => selects.push(id) });
  env.frame(); view.setData(data(3000)); env.settle();
  env.clock += 1500;                                                    // an occasional update
  view.setData(data(3000, { seed: 2, shift: 1 }));                     // every point flies (900 ms)
  for (let k = 0; k < 20; k++) env.frame();
  const drawn = (i, t) => {
    const x = view.from[4 * i] + (view.to[4 * i] - view.from[4 * i]) * t - view.center[0], y = view.from[4 * i + 1] + (view.to[4 * i + 1] - view.from[4 * i + 1]) * t - view.center[1];
    const scale = view.height * 1.20710678 / view.distance;
    return { x: view.width / 2 + x * scale, y: view.height / 2 - y * scale };
  };
  const t = view.transition(env.clock), builds = view.gridBuilds, scans = view.drawnPicks;
  assert.ok(t > .1 && t < .9, `t ${t}`);
  let clicks = 0, wrong = 0, elsewhere = 0;
  for (let i = 0; i < view.count; i += 11) {
    const p = drawn(i, t);
    if (!(view.from[4 * i + 3] > 0 && view.to[4 * i + 3] > 0) || p.x < 6 || p.y < 6 || p.x > view.width - 6 || p.y > view.height - 6) continue;
    const at = { clientX: RECT.left + p.x, clientY: RECT.top + p.y, pointerId: 1 };
    env.canvas.emit('pointerdown', { ...at, button: 0 }); env.canvas.emit('pointerup', at); clicks++;
    const got = selects.at(-1), q = got == null ? null : drawn(got, t);
    if (!q || Math.hypot(q.x - p.x, q.y - p.y) > .5) wrong++;           // not the cell drawn under the pointer
    if (got !== scanPick(view, p.x, p.y)) elsewhere++;                   // the targets would have given another cell
  }
  assert.ok(clicks > 150, `${clicks} clicks`); assert.equal(wrong, 0);
  assert.ok(elsewhere > clicks / 2, `${elsewhere} of ${clicks}`);
  assert.equal(view.gridBuilds, builds);                                // moving points: no grid of stale positions
  assert.equal(view.drawnPicks, scans + clicks);                        // one scan per click
  const next = view.transition(env.clock + 16), p = drawn(1500, next);
  move(env, p); move(env, p); env.frame();                              // the pointer moves onto a moving point
  const h = view.hover, q = h == null ? null : drawn(h, next);
  assert.ok(q && Math.hypot(q.x - p.x, q.y - p.y) <= .5, `hovered ${h}`); assert.equal(hovers.at(-1), h);
  assert.equal(view.gridBuilds, builds); assert.equal(view.drawnPicks, scans + clicks + 1);   // once per frame, not per event
  env.settle();                                                         // the morph ends under the resting pointer
  assert.equal(view.gridBuilds, builds + 1); assert.equal(view.hover, scanPick(view, p.x, p.y));
  assert.equal(view.drawnPicks, scans + clicks + 1);
  const settled = hovers.length; env.clock += 1500; view.setData(data(3000, { seed: 3, shift: 2 }));
  for (let k = 0; k < 5; k++) env.frame();
  assert.equal(hovers.length, settled);                                 // a resting pointer: no hover churn while morphing
  env.settle(); assert.equal(view.hover, scanPick(view, p.x, p.y));
});

test('labels keep one node per cluster across updates and never read layout', () => {
  const env = browser(), view = new UmapView(env.canvas);
  env.frame();
  const labels = [{ id: 0, text: 'Alpha' }, { id: 1, text: 'Beta' }, { id: 4, text: 'Gamma' }];
  view.setData({ ...data(4000), labels }); env.settle();
  const nodes = labelsOf(view), reads = env.layoutReads, created = env.created;
  assert.equal(nodes.length, 3);
  assert.deepEqual(nodes.map(n => n.childNodes[1].textContent), ['Alpha', 'Beta', 'Gamma']);
  assert.ok(nodes.every(n => n.style.display === 'block' && /^translate\(-?\d+px,-?\d+px\)$/.test(n.style.transform)));
  assert.equal(nodes[0].childNodes[0].style.background, 'rgb(0,114,178)');
  view.setData({ ...data(4000, { seed: 2, shift: .3 }), labels: [labels[0], { id: 1, text: 'Beta 2' }, labels[2]] }); env.settle();
  assert.deepEqual(labelsOf(view), nodes); assert.equal(env.created, created);   // same nodes, nothing rebuilt
  assert.equal(nodes[1].childNodes[1].textContent, 'Beta 2');
  assert.equal(env.layoutReads, reads);                                // measured with text metrics, placed without reads
  const writes = env.styleWrites;
  view.select(17); env.frame();                                        // a redraw that moves nothing writes no label style
  assert.equal(env.styleWrites, writes);
  view.highlightCluster(1); env.frame();
  assert.deepEqual(nodes.map(n => n.style.opacity), ['.25', '1', '.25']);
  assert.equal(env.styleWrites, writes + 2);                           // relaid out: only the two opacities changed
  view.hideClusters(new Set([4])); env.frame();
  assert.equal(nodes[2].style.display, 'none');
  view.setData({ ...data(4000, { seed: 3 }), labels: [labels[2], labels[0]], centroids: new Map([[0, [1, 2, 0]], [4, [5, 6]]]) }); env.settle();
  assert.deepEqual(labelsOf(view), [nodes[0], nodes[2]]); assert.equal(nodes[1].parentElement, null);
  assert.deepEqual(view.centroids.get(0), [1, 2, 0]); assert.deepEqual(view.centroids.get(4), [5, 6, 0]);
  assert.deepEqual(view.labels.map(l => l.id), [4, 0]);
  assert.equal(env.layoutReads, reads);
});

test('setData uploads changed buffers in place; unchanged emphasis neither uploads nor redraws', () => {
  const env = browser(), view = new UmapView(env.canvas);
  env.frame(); view.setData(data(30000)); env.settle();
  assert.equal(env.calls('bufferData'), 5); assert.equal(env.calls('bufferSubData'), 0);
  assert.equal(env.calls('vertexAttribPointer'), 5);                    // attribute layout set once, in the VAO
  view.setData(data(30000, { seed: 2 })); env.settle();
  // A flight draws mix(from, to): positions and colours, from and to, rewritten in place; weights unchanged: not sent.
  assert.equal(env.calls('bufferData'), 5); assert.equal(env.calls('bufferSubData'), 4);
  view.highlightCluster(2); assert.equal(env.calls('bufferSubData'), 5); env.settle();
  assert.equal(view.weights[2], 1); assert.equal(view.weights[3], Math.fround(.12));
  const sub = env.calls('bufferSubData'), clears = env.calls('clear');
  view.highlightCluster(2); view.highlightCells(null); view.select(null);
  assert.equal(env.frames.size, 0); assert.equal(env.calls('bufferSubData'), sub);
  view.highlightCluster(undefined); view.highlightCells([3, 9]); env.settle();
  assert.equal(view.weights[3], 1); assert.equal(view.weights[4], Math.fround(.12)); assert.equal(env.calls('clear'), clears + 1);
  view.highlightCells(null); view.hideClusters(new Set([1])); env.settle();
  assert.equal(view.weights[1], Math.fround(.12)); assert.equal(view.weights[2], 1);
  view.setData(data(31000)); assert.equal(env.calls('bufferData'), 10);   // a larger map reallocates
  view.canvas.emit('webglcontextlost'); assert.equal(env.frames.size, 0);
  view.canvas.emit('webglcontextrestored'); assert.equal(env.calls('bufferData'), 15); assert.equal(env.frames.size, 1);
});

test('a 40k-cell update sorts at most a 4096-point sample, and nothing when the worker sends bounds and centroids', () => {
  const env = browser(), view = new UmapView(env.canvas), labels = [0, 1, 2, 3, 4, 5].map(id => ({ id, text: `C${id}` }));
  env.frame();
  const lengths = sortLengths(() => view.setData({ ...data(40000), labels }));
  assert.ok(lengths.length && Math.max(...lengths) <= 4096, `sorted ${Math.max(...lengths)}`);
  env.settle();
  const centroids = new Map(labels.map(l => [l.id, [l.id * 3 + 1, (l.id % 3) * 2 + 1, 0]])), bounds = { x: [-1, 17], y: [0, 6] };
  assert.deepEqual(sortLengths(() => view.setData({ ...data(40000, { seed: 5 }), labels, centroids, bounds })), []);
  assert.deepEqual(view.center, [8, 3, 0]);                             // fitted to the given bounds
  assert.deepEqual(view.centroids.get(5), [16, 5, 0]);
  view.userCamera = true; view.center = [0, 0, 0];
  view.setData({ ...data(40000, { seed: 6 }), labels, centroids, bounds: { x: [0, 2], y: [0, 2] } });
  assert.deepEqual(view.center, [0, 0, 0]);                             // the user's camera is kept
  view.canvas.emit('dblclick'); assert.deepEqual(view.center, [1, 1, 0]);   // double-click hands it back
});

test('public methods keep working and dispose releases everything', () => {
  const env = browser(), selects = [], view = new UmapView(env.canvas, { onSelect: id => selects.push(id) });
  env.frame(); view.setData({ ...data(3000), labels: [{ id: 0, text: 'A' }] }); env.settle();
  view.setAutoRotate(true); assert.equal(view.autoRotate, false);
  view.select(12); assert.equal(view.selected, 12); env.frame();
  const center = view.center; view.pan(40, 0); assert.notDeepEqual(view.center, center);
  view.fit(); assert.deepEqual(view.center, center);
  const reads = env.rectReads;
  env.canvas.rect = { ...RECT, width: 200 }; view.resize(); env.frame();
  assert.equal(env.rectReads, reads + 1); assert.equal(view.width, 200); assert.equal(env.canvas.width, 200);
  assert.equal(view.overlay.style.width, '200px');
  env.canvas.rect = { ...RECT, width: 0, height: 0 }; env.observers[0].fn(); env.frame();
  assert.equal(view.overlay.style.display, 'none');                     // a collapsed canvas hides its labels
  env.canvas.rect = RECT; env.observers[0].fn(); env.frame();
  assert.equal(view.overlay.style.display, 'block');
  view.dispose();
  assert.equal(view.overlay.parentElement, null); assert.equal(env.frames.size, 0); assert.equal(env.observers[0].disconnected, true);
  assert.equal([...env.canvas.listeners.values()].flat().length, 0); assert.equal([...window.listeners.values()].flat().length, 0);
  assert.equal(env.calls('deleteBuffer'), 5);
  view.resize(); view.setData(data(10)); assert.equal(env.frames.size, 0);
});
