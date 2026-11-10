import { clusterColor } from './clusters.js';
import { harness } from './harness-hook.mjs';

export function median(values) {
  const sorted = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b), n = sorted.length;
  return n ? (sorted[(n - 1) >> 1] + sorted[n >> 1]) / 2 : NaN;
}
const finite3 = (a, j) => Number.isFinite(a[j]) && Number.isFinite(a[j + 1]) && Number.isFinite(a[j + 2]);
const middle = s => { const n = s.length; return n ? (s[(n - 1) >> 1] + s[n >> 1]) / 2 : NaN; };
/** Coordinate-wise medians per cluster (ids in order of first appearance), excluding the entire point if any coordinate
 * is invalid. Each median uses an evenly spaced, deterministic subsample of at most `limit` points of its cluster (exact
 * for smaller clusters): linear in the number of points, never a sort of all of them. */
export function clusterCentroids(clusters, coords, limit = 512) {
  let max = -1, total = 0;
  for (let i = 0; i < clusters.length; i++) if (clusters[i] > max) max = clusters[i];
  const counts = new Int32Array(max + 1), size = new Int32Array(max + 1), offset = new Int32Array(max + 1), seen = new Int32Array(max + 1), taken = new Int32Array(max + 1), order = [];
  for (let i = 0; i < clusters.length; i++) { const c = clusters[i]; if (c >= 0 && finite3(coords, 3 * i) && counts[c]++ === 0) order.push(c); }
  for (const c of order) { size[c] = Math.min(counts[c], Math.max(1, limit)); offset[c] = total; total += size[c]; }
  const axes = [new Float64Array(total), new Float64Array(total), new Float64Array(total)];
  for (let i = 0; i < clusters.length; i++) {
    const c = clusters[i];
    if (c < 0 || !finite3(coords, 3 * i)) continue;
    const k = taken[c];
    if (k < size[c] && seen[c] === Math.floor(k * counts[c] / size[c])) { for (let a = 0; a < 3; a++) axes[a][offset[c] + k] = coords[3 * i + a]; taken[c] = k + 1; }
    seen[c]++;
  }
  return new Map(order.map(c => [c, axes.map(a => middle(a.subarray(offset[c], offset[c] + size[c]).sort()))]));
}
/** 2nd-98th percentile x/y bounds of the valid (w > 0) points of an xyzw array, from an evenly spaced deterministic
 * subsample of at most `limit` points (exact when there are fewer). Null when no point is valid. */
export function fitBounds(points, count, limit = 4096, lo = .02, hi = .98) {
  let valid = 0;
  for (let i = 0; i < count; i++) if (points[4 * i + 3] > 0) valid++;
  if (!valid) return null;
  const size = Math.min(valid, limit), xs = new Float64Array(size), ys = new Float64Array(size);
  for (let i = 0, rank = 0, k = 0; i < count && k < size; i++) {
    if (!(points[4 * i + 3] > 0)) continue;
    if (rank++ === Math.floor(k * valid / size)) { xs[k] = points[4 * i]; ys[k++] = points[4 * i + 1]; }
  }
  xs.sort(); ys.sort();
  const q = (a, f) => a[Math.min(size - 1, Math.max(0, Math.round(f * (size - 1))))];
  return { x: [q(xs, lo), q(xs, hi)], y: [q(ys, lo), q(ys, hi)] };
}
const validBounds = b => b && [b.x?.[0], b.x?.[1], b.y?.[0], b.y?.[1]].every(Number.isFinite) && b.x[1] >= b.x[0] && b.y[1] >= b.y[0] ? b : null;

// Hover/click picking on a settled map: a uniform screen-space grid of the projected TARGET positions. The cell is at
// least the largest hover radius (6 px), so a query only visits the cells around the pointer instead of every point.
// While a morph runs the points are drawn between their old and new positions, so a pick scans those (pickDrawn).
const PICK_CELL = 12, PICK_REACH = 6;
/** Grid of the points of an xyzw array (w >= .15) as the shader projects them for view {center, distance, width,
 * height, basis}. Points outside the canvas or the depth range are left out. `old` is reused when the sizes match. */
export function buildPickGrid(points, count, { center, distance, width, height, basis: { right: r, up: u, back: b } }, old = null) {
  const cols = Math.max(1, Math.ceil(width / PICK_CELL)), rows = Math.max(1, Math.ceil(height / PICK_CELL)), cells = cols * rows;
  const g = old && old.px.length === count && old.start.length === cells + 1 ? old
    : { px: new Float64Array(count), py: new Float64Array(count), depth: new Float64Array(count), cellOf: new Int32Array(count), items: new Int32Array(count), start: new Int32Array(cells + 1), fill: new Int32Array(cells), hits: new Int32Array(64) };
  Object.assign(g, { cols, rows, distance, count });
  const { px, py, depth, cellOf, items, start, fill } = g, [cx, cy, cz] = center, near = distance * .001, far = distance * 100, unit = height * 1.20710678;
  start.fill(0);
  for (let i = 0; i < count; i++) {
    cellOf[i] = -1;
    if (!(points[4 * i + 3] >= .15)) continue;
    const x = points[4 * i] - cx, y = points[4 * i + 1] - cy, z = points[4 * i + 2] - cz, d = distance - x * b[0] - y * b[1] - z * b[2];
    if (!(d > near && d < far)) continue;
    const scale = unit / d, sx = width / 2 + (x * r[0] + y * r[1] + z * r[2]) * scale, sy = height / 2 - (x * u[0] + y * u[1] + z * u[2]) * scale;
    if (!(sx >= 0 && sy >= 0 && sx <= width && sy <= height)) continue;
    px[i] = sx; py[i] = sy; depth[i] = d;
    const k = Math.min(rows - 1, Math.floor(sy / PICK_CELL)) * cols + Math.min(cols - 1, Math.floor(sx / PICK_CELL));
    cellOf[i] = k; start[k + 1]++;
  }
  for (let k = 0; k < cells; k++) { start[k + 1] += start[k]; fill[k] = start[k]; }
  for (let i = 0; i < count; i++) if (cellOf[i] >= 0) items[fill[cellOf[i]]++] = i;
  return g;
}
/** The point under (x, y) CSS px: within its marker radius, nearest first, the front one among near-ties (the same rule,
 * in the same index order, as a scan of every point). Points of `hidden` clusters are skipped. Null when none. */
export function pickNearest(grid, x, y, clusters = null, hidden = null) {
  if (!grid) return null;
  const { cols, rows, start, items, px, py, depth, distance } = grid;
  const c0 = Math.max(0, Math.floor((x - PICK_REACH) / PICK_CELL)), c1 = Math.min(cols - 1, Math.floor((x + PICK_REACH) / PICK_CELL));
  const r0 = Math.max(0, Math.floor((y - PICK_REACH) / PICK_CELL)), r1 = Math.min(rows - 1, Math.floor((y + PICK_REACH) / PICK_CELL));
  let hits = grid.hits, n = 0;
  for (let row = r0; row <= r1; row++) for (let col = c0; col <= c1; col++) {
    const k = row * cols + col;
    for (let j = start[k]; j < start[k + 1]; j++) {
      const i = items[j];
      if (hidden?.has(clusters?.[i]) || Math.hypot(px[i] - x, py[i] - y) > Math.max(5, Math.min(12, 4 * distance / depth[i]) / 2)) continue;
      if (n === hits.length) { const grown = new Int32Array(2 * n); grown.set(hits); hits = grid.hits = grown; }
      hits[n++] = i;
    }
  }
  const order = hits.subarray(0, n).sort();
  let best = null, bestScore = Infinity, bestDepth = Infinity;
  for (let j = 0; j < n; j++) {
    const i = order[j], d = Math.hypot(px[i] - x, py[i] - y);
    if (d < bestScore - .5 || Math.abs(d - bestScore) <= .5 && depth[i] < bestDepth) { best = i; bestScore = d; bestDepth = depth[i]; }
  }
  return best;
}
/** The point under (x, y) as the shader draws it during a morph, at mix(from, to, t) (w >= .15), by pickNearest's rule.
 * One pass over every point (a grid of moving points would be stale a frame later): a click, or a hover after the
 * pointer moved, while the map morphs. */
export function pickDrawn(from, to, t, count, x, y, { center, distance, width, height, basis: { right: r, up: u, back: b } }, clusters = null, hidden = null) {
  const [cx, cy, cz] = center, near = distance * .001, far = distance * 100, unit = height * 1.20710678;
  let best = null, bestScore = Infinity, bestDepth = Infinity;
  for (let i = 0, q = 0; i < count; i++, q += 4) {
    const px = from[q] + (to[q] - from[q]) * t - cx, py = from[q + 1] + (to[q + 1] - from[q + 1]) * t - cy, pz = from[q + 2] + (to[q + 2] - from[q + 2]) * t - cz;
    const d = distance - px * b[0] - py * b[1] - pz * b[2];
    if (!(d > near && d < far)) continue;
    const scale = unit / d, sx = width / 2 + (px * r[0] + py * r[1] + pz * r[2]) * scale, sy = height / 2 - (px * u[0] + py * u[1] + pz * u[2]) * scale;
    if (!(Math.abs(sx - x) <= PICK_REACH && Math.abs(sy - y) <= PICK_REACH)) continue;   // most points: beyond any marker
    if (!(from[q + 3] + (to[q + 3] - from[q + 3]) * t >= .15) || hidden?.has(clusters?.[i])) continue;
    if (!(sx >= 0 && sy >= 0 && sx <= width && sy <= height)) continue;
    const dist = Math.hypot(sx - x, sy - y);
    if (dist > Math.max(5, Math.min(12, 4 * distance / d) / 2)) continue;
    if (dist < bestScore - .5 || Math.abs(dist - bestScore) <= .5 && d < bestDepth) { best = i; bestScore = dist; bestDepth = d; }
  }
  return best;
}

/** In-place greedy placement: items {ax, ay (anchor), width, height} get {x, y, placed}; earlier items win. */
export function placeLabels(items, width, height, gap = 4) {
  for (let i = 0; i < items.length; i++) {
    const l = items[i]; l.placed = false;
    if (!(Number.isFinite(l.ax) && Number.isFinite(l.ay) && Number.isFinite(l.width) && Number.isFinite(l.height))) continue;
    l.x = Math.max(0, Math.min(width - l.width, l.ax - l.width / 2));
    for (let attempt = 0; attempt < 4 && !l.placed; attempt++) {
      l.y = l.ay + (attempt === 0 ? -l.height - 8 : attempt === 1 ? 8 : attempt === 2 ? -2 * l.height - 12 : l.height + 12);
      if (l.x < 0 || l.y < 0 || l.x + l.width > width || l.y + l.height > height) continue;
      l.placed = true;
      for (let j = 0; j < i; j++) { const p = items[j]; if (p.placed && l.x < p.x + p.width + gap && l.x + l.width + gap > p.x && l.y < p.y + p.height + gap && l.y + l.height + gap > p.y) { l.placed = false; break; } }
    }
  }
  return items;
}
// Label boxes are sized from canvas text metrics (11px system-ui; 5 px padding, 6 px dot + 5 px gap, 220 px max, 14 px
// line + 6 px padding), so labels are never measured through layout.
const LABEL_FONT = '11px system-ui', LABEL_HEIGHT = 20;
const LABEL_CSS = 'position:absolute;left:0;top:0;display:none;white-space:nowrap;padding:3px 5px;border-radius:4px;background:#0b0e13bb;max-width:220px;overflow:hidden;text-overflow:ellipsis;line-height:14px';
let metrics;
function textWidth(text) {
  if (metrics === undefined) try { metrics = (typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(1, 1) : document.createElement('canvas')).getContext('2d') || null; if (metrics) metrics.font = LABEL_FONT; } catch { metrics = null; }
  return metrics ? metrics.measureText(text).width : text.length * 6.2;
}
export const labelWidth = (text, measure = textWidth) => Math.min(220, Math.ceil(measure(text)) + 21);
const paletteRGB = (palette, id) => (typeof palette === 'function' ? palette(id) : palette?.[id]) || clusterColor(id);
const css = rgb => `rgb(${Math.round(rgb[0] * 255)},${Math.round(rgb[1] * 255)},${Math.round(rgb[2] * 255)})`;

const ease = t => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
const mix = (a, b, t) => a + (b - a) * t;
// Every new layout flies: each dot travels from where it is drawn to its new place, which is how the map shows what
// changed (better than layouts that just appear, at any map size). While layouts stream in (a live run: a
// whole sample about once a second, a region every 0.1-0.5 s) a flight fills FLIGHT_FILL of the time since the previous
// one, so the dots keep moving and each flight starts from where they are; an occasional layout (a step, a paused run,
// a new setting) flies for FLIGHT_MS. Measured, not assumed.
export const FLIGHT_MS = 900, FLIGHT_MIN_MS = 200, FLIGHT_FILL = .9;
/** Milliseconds a new layout flies, `interval` ms after the previous one started flying. */
export const morphDuration = (interval = Infinity) => Math.max(FLIGHT_MIN_MS, Math.min(FLIGHT_MS, FLIGHT_FILL * interval));
const DIM = Math.fround(.12);
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec4 aFrom;
layout(location=1) in vec4 aTo;
layout(location=2) in vec3 aColorFrom;
layout(location=3) in vec3 aColorTo;
layout(location=4) in float aEmphasis;
uniform float uMix, uDistance, uAspect, uRatio, uSelected, uHover;
uniform vec3 uCenter, uRight, uUp, uBack;
// Go deeper (qc-deeper.js, ringCells): up to 6 rings, uRings the points, uRingKinds 1 the window's cell (white), 2 a
// neighbour its RNA seems to come from (magenta), 3 one that looks like its own kind (grey); none while uRingCount is 0.
uniform float uRings[6], uRingKinds[6];
uniform int uRingCount;
out vec4 vColor;
out float vRing;
out float vSource;
void main() {
  vec4 p = mix(aFrom,aTo,uMix);
  vec3 d = p.xyz-uCenter;
  float z = uDistance-dot(d,uBack);
  float nearZ = uDistance*.001, farZ = uDistance*100.;
  gl_Position = vec4(2.41421356*dot(d,uRight)/uAspect, 2.41421356*dot(d,uUp),
    (farZ+nearZ)/(farZ-nearZ)*z-2.*farZ*nearZ/(farZ-nearZ),z);
  vRing = float(float(gl_VertexID)==uSelected);
  vSource = 0.;
  for (int k = 0; k < 6; k++) { if (k >= uRingCount) break; if (float(gl_VertexID) == uRings[k]) vSource = uRingKinds[k]; }
  float hover = float(float(gl_VertexID)==uHover);
  gl_PointSize = (clamp(4.*uDistance/max(z,nearZ),2.,12.)+max(vRing, step(.5, vSource))*6.+hover*2.)*uRatio;
  float shade = clamp(.9 + .12*dot(d,uBack)/uDistance,.65,1.);
  vColor = vec4(mix(aColorFrom,aColorTo,uMix)*shade, p.w*aEmphasis);
}`;
const FS = `#version 300 es
precision highp float;
in vec4 vColor;
in float vRing;
in float vSource;
out vec4 outColor;
void main() {
  float r = length(gl_PointCoord*2.-1.);
  if(r>1. || vColor.a<.005) discard;
  float edge = 1.-smoothstep(.85,1.,r);
  vec3 color = vColor.rgb;
  float alpha = vColor.a;
  if(vSource>.5 && vRing<.5 && r>.6) { if(r<.72) discard; color = vSource>2.5 ? vec3(.604,.643,.698) : vSource>1.5 ? vec3(.835,.318,.506) : vec3(1.); alpha = 1.; }
  if(vRing>.5 && r>.6) { if(r<.72) discard; color=vec3(1.); }
  outColor=vec4(color,alpha*edge);
}`;
function program(gl) {
  const p = gl.createProgram();
  try {
    for (const [type, source] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, FS]]) {
      const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
      const ok = gl.getShaderParameter(s, gl.COMPILE_STATUS), log = gl.getShaderInfoLog(s);
      if (ok) gl.attachShader(p, s);
      gl.deleteShader(s);
      if (!ok) throw new Error(log);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  } catch (error) { gl.deleteProgram(p); throw error; }
}

/** Standalone perspective WebGL2 view. Palette is an RGB array indexed by cluster id,
 * or (clusterId) => RGB. Callers map summary.cluster.color to that palette themselves.
 * No fetches: decode unaligned binary coords using DataView when ncells is odd.
 * Renders on demand: a frame is requested only by a change (data, morph, camera, size, hover, selection, emphasis) and
 * the view is idle otherwise. Hover is picked once per animation frame after pointer moves, through a screen-space grid
 * of the target positions that is rebuilt only when the data, the camera or the canvas size changed. While a morph
 * runs, a click or a moved pointer picks among the points as drawn (one scan), and hover is picked again when the
 * morph ends, so it names the point under a resting pointer once the map has settled. Every new layout morphs (flies,
 * morphDuration), and the view is idle again once the flight has landed. Covered by Compare's full-screen view
 * (setCovered), it draws nothing at all; uncovered, what is pending is drawn.
 */
export class UmapView {
  constructor(canvas, { onHover = () => {}, onSelect = () => {} } = {}) {
    this.canvas = canvas; this.onHover = onHover; this.onSelect = onSelect;
    this.center = [0, 0, 0]; this.distance = 10; this.yaw = 0; this.pitch = 0;   // 2-D: always face-on, no orbit
    this.count = 0; this.selected = null; this.hover = null; this.highlight = null; this.clusterHighlight = null;
    this.listeners = []; this.started = -Infinity; this.duration = FLIGHT_MS; this.labels = []; this.labelNodes = new Map(); this.centroids = new Map();
    this.covered = false;   // a full-screen view over the page (setCovered): no frame asked for meanwhile
    this.version = this.cameraVersion = this.gridBuilds = this.drawnPicks = this.drawCount = 0;
    // The overlay sits in the canvas's own container, so labels follow the canvas through any layout change without
    // reading its position every frame.
    this.overlay = document.createElement('div');
    this.overlay.style.cssText = 'position:absolute;left:0;top:0;width:0;height:0;display:none;pointer-events:none;overflow:hidden;contain:strict;z-index:10;font:11px system-ui;color:#e6edf3';
    (canvas.parentElement || document.body).append(this.overlay);
    this.gizmo = document.createElement('div'); this.overlay.append(this.gizmo);
    this.gizmo.style.cssText = 'position:absolute;left:36px;bottom:36px;opacity:.5';
    this.axes = ['X', 'Y', 'Z'].map((text, i) => {
      const node = document.createElement('span'); node.textContent = text;
      node.style.cssText = `position:absolute;transform-origin:0 50%;width:25px;border-top:1px solid ${['#e69f00', '#56b4e9', '#cc79a7'][i]};font-size:9px;text-align:right`;
      this.gizmo.append(node); return node;
    });
    this.oldTouchAction = canvas.style.touchAction; canvas.style.touchAction = 'none';
    this.listen(canvas, 'webglcontextlost', e => { e.preventDefault(); harness?.count('contextLosses'); this.lost = true; cancelAnimationFrame(this.raf); this.raf = null; this.overlay.style.display = 'none'; this.box = null; });
    this.listen(canvas, 'webglcontextrestored', () => { this.lost = false; this.init(); this.upload(); this.dirty = this.labelsDirty = this.sizeStale = true; this.wake(); });
    try { this.init(); } catch (error) {
      this.listeners.forEach(fn => fn()); this.overlay.remove();
      canvas.style.touchAction = this.oldTouchAction; throw error;
    }
    this.events();
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(canvas);
    this.listen(window, 'scroll', () => { this.rectStale = true; }, true);   // only pointer coordinates depend on it
    this.listen(window, 'resize', () => this.resize());
    this.resize();
  }
  listen(target, type, fn, options) { target.addEventListener(type, fn, options); this.listeners.push(() => target.removeEventListener(type, fn, options)); }
  init() {
    const gl = this.canvas.getContext('webgl2', { alpha: false, antialias: true });
    if (!gl) throw new Error('UMAP requires WebGL2');
    this.gl = gl; this.program = program(gl); this.vao = gl.createVertexArray();
    this.buffers = Array.from({ length: 5 }, () => gl.createBuffer()); this.uploaded = [-1, -1, -1, -1, -1];
    gl.bindVertexArray(this.vao);   // attribute layout is VAO state: set once, not per upload
    this.buffers.forEach((buffer, i) => { gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(i); gl.vertexAttribPointer(i, [4,4,3,3,1][i], gl.FLOAT, false, 0, 0); });
    gl.bindVertexArray(null);
    this.uniforms = Object.fromEntries(['Mix','Distance','Aspect','Ratio','Selected','Hover','Center','Right','Up','Back','Rings','RingKinds','RingCount'].map(n => [n, gl.getUniformLocation(this.program, `u${n}`)]));
  }
  /** Re-measure the canvas in the next frame (layout is read once there, before any write). */
  resize() { this.sizeStale = this.rectStale = true; this.wake(); }
  measure() {
    this.sizeStale = false;
    const r = this.canvas.getBoundingClientRect(), left = this.canvas.offsetLeft, top = this.canvas.offsetTop, dpr = window.devicePixelRatio || 1;
    this.rect = r; this.rectStale = false;
    const box = `${left},${top},${r.width},${r.height}`;
    if (box !== this.box) {
      const s = this.overlay.style; this.box = box;
      s.left = `${left}px`; s.top = `${top}px`; s.width = `${r.width}px`; s.height = `${r.height}px`; s.display = r.width && r.height ? 'block' : 'none';
    }
    if (r.width === this.width && r.height === this.height && dpr === this.ratio) return;
    this.width = r.width; this.height = r.height; this.ratio = dpr;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr)); this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    if (!this.fitted && this.count && r.width && r.height) this.fit();
    this.cameraChanged(); this.requestPick();
  }
  clientRect() {
    if (this.rectStale || !this.rect) { this.rect = this.canvas.getBoundingClientRect(); this.rectStale = false; }
    return this.rect;
  }
  basis() {
    const c = Math.cos(this.yaw), s = Math.sin(this.yaw), p = Math.cos(this.pitch), q = Math.sin(this.pitch);
    const b=this.cachedBasis ||= {right:new Float32Array(3),up:new Float32Array(3),back:new Float32Array(3)};
    b.right[0]=c;b.right[1]=0;b.right[2]=-s;b.up[0]=-s*q;b.up[1]=p;b.up[2]=-c*q;b.back[0]=s*p;b.back[1]=q;b.back[2]=c*p;return b;
  }
  transition(time = performance.now()) { return this.duration > 0 ? ease((time - this.started) / this.duration) : 1; }
  setData({ clusters, coords, palette = clusterColor, labels = [], centroids = null, bounds = null }) {
    if (!(clusters instanceof Int16Array) || !(coords instanceof Float32Array) || coords.length !== clusters.length * 3) throw new TypeError('Expected Int16Array(n) clusters and Float32Array(3n) coords');
    const now = performance.now(), t = this.transition(now);
    const m = clusters.length, old = this.count, n = Math.max(m, old), reuse = n === old && !!this.from;
    const of = this.from, ot = this.to, ocf = this.colorFrom, oct = this.colorTo;
    // Same size (every update of a run): rewrite in place, element i only reads its own previous values.
    const from = reuse ? of : new Float32Array(n * 4), to = reuse ? ot : new Float32Array(n * 4), cf = reuse ? ocf : new Float32Array(n * 3), ct = reuse ? oct : new Float32Array(n * 3);
    let max = -1;
    for (let i = 0; i < m; i++) if (clusters[i] > max) max = clusters[i];
    const table = new Float32Array(3 * (max + 1)), known = new Uint8Array(max + 1);   // palette looked up once per cluster
    for (let i = 0; i < n; i++) {
      const c = i < m ? clusters[i] : -1, j = 3 * i, q = 4 * i, valid = c >= 0 && finite3(coords, j);
      let a = 0, x = 0, y = 0, z = 0, r = 0, g = 0, b = 0, pr = 0, pg = 0, pb = 0;
      if (i < old) {
        a = mix(of[q + 3], ot[q + 3], t); x = mix(of[q], ot[q], t); y = mix(of[q + 1], ot[q + 1], t); z = mix(of[q + 2], ot[q + 2], t);
        r = mix(ocf[j], oct[j], t); g = mix(ocf[j + 1], oct[j + 1], t); b = mix(ocf[j + 2], oct[j + 2], t);
      }
      if (valid) {
        if (!known[c]) { const rgb = paletteRGB(palette, c); table[3 * c] = rgb[0]; table[3 * c + 1] = rgb[1]; table[3 * c + 2] = rgb[2]; known[c] = 1; }
        pr = table[3 * c]; pg = table[3 * c + 1]; pb = table[3 * c + 2];
      }
      const keep = a > 0 || !valid, shown = a > 0;
      from[q] = keep ? x : coords[j]; from[q + 1] = keep ? y : coords[j + 1]; from[q + 2] = keep ? z : coords[j + 2]; from[q + 3] = a;
      to[q] = valid ? coords[j] : from[q]; to[q + 1] = valid ? coords[j + 1] : from[q + 1]; to[q + 2] = valid ? coords[j + 2] : from[q + 2]; to[q + 3] = valid ? 1 : 0;
      cf[j] = shown ? r : pr; cf[j + 1] = shown ? g : pg; cf[j + 2] = shown ? b : pb;
      ct[j] = valid ? pr : cf[j]; ct[j + 1] = valid ? pg : cf[j + 1]; ct[j + 2] = valid ? pb : cf[j + 2];
    }
    this.from = from; this.to = to; this.colorFrom = cf; this.colorTo = ct; this.count = n; this.version++;
    if (!reuse) this.clusters = new Int16Array(n);
    this.clusters.fill(-1); this.clusters.set(clusters);
    // Fly from the state on screen; the hover under a resting pointer is picked again when the flight lands (draw).
    this.duration = morphDuration(now - this.started); this.started = now; this.morphing = true;
    this.bounds = validBounds(bounds);
    this.centroids = this.anchors(clusters, coords, centroids, labels);
    this.syncLabels(labels, palette, t);
    // A flight draws mix(from, to): both the start and the target positions and colours are uploaded.
    this.upload((this.weigh() ? 16 : 0) | 15);
    // Keep the whole embedding in view as it evolves, until the user takes the camera (double-click hands it back).
    if ((!this.fitted || !this.userCamera) && this.width && this.height) this.fit();
    // The morph starts from the points on screen, so the hovered point is still the one under the pointer: hover is
    // picked again when the morph ends (draw), not for every update.
    this.dirty = this.labelsDirty = true; this.wake();
  }
  /** Label anchors: the caller's centroids (summary.clusters[].centroid) when given, else subsampled medians. */
  anchors(clusters, coords, centroids, labels) {
    const anchors = new Map();
    if (!labels.length) return anchors;
    if (centroids instanceof Map) for (const [id, v] of centroids) if (v && Number.isFinite(v[0]) && Number.isFinite(v[1])) anchors.set(id, [+v[0], +v[1], Number.isFinite(v[2]) ? +v[2] : 0]);
    if (labels.some(l => !anchors.has(l.id))) for (const [id, v] of clusterCentroids(clusters, coords)) if (!anchors.has(id)) anchors.set(id, v);
    return anchors;
  }
  /** One DOM node per cluster id, kept across updates; text is re-measured (canvas metrics) only when it changed. */
  syncLabels(labels, palette, t) {
    const next = [], keep = new Set();
    for (const l of labels) {
      const target = this.centroids.get(l.id);
      if (!target || keep.has(l.id)) continue;
      keep.add(l.id);
      let rec = this.labelNodes.get(l.id);
      if (!rec) {
        const node = document.createElement('div'), dot = document.createElement('span'), text = document.createTextNode('');
        node.style.cssText = LABEL_CSS; dot.style.cssText = 'display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:5px';
        node.append(dot, text); this.overlay.append(node);
        rec = { id: l.id, node, dot, textNode: text, text: null, color: null, width: 0, height: LABEL_HEIGHT, from: target, x: 0, y: 0, ax: NaN, ay: NaN, placed: false, shown: false, left: NaN, top: NaN, opacity: null };
        this.labelNodes.set(l.id, rec);
      } else rec.from = rec.target.map((v, k) => mix(rec.from[k], v, t));   // from where the label is on screen now
      rec.target = target;
      const text = String(l.text ?? ''), color = css(paletteRGB(palette, l.id));
      if (text !== rec.text) { rec.textNode.textContent = rec.text = text; rec.width = labelWidth(text); }
      if (color !== rec.color) rec.dot.style.background = rec.color = color;
      next.push(rec);
    }
    for (const [id, rec] of this.labelNodes) if (!keep.has(id)) { rec.node.remove(); this.labelNodes.delete(id); }
    this.labels = next;
  }
  /** Writes the per-point emphasis into this.weights; true when anything changed. */
  weigh() {
    const n = this.count, clusters = this.clusters || new Int16Array(0);
    let w = this.weights, changed = false, max = -1, mask = null;
    if (!w || w.length !== n) { w = this.weights = new Float32Array(n); changed = true; }
    for (let i = 0; i < n; i++) if (clusters[i] > max) max = clusters[i];
    const table = new Float32Array(max + 2);   // index cluster + 1 (-1 = not analysed)
    for (let c = -1; c <= max; c++) table[c + 1] = !this.hiddenClusters?.has(c) && (this.clusterHighlight === null || c === this.clusterHighlight) ? 1 : DIM;
    if (this.highlight) { mask = new Uint8Array(n); for (const id of this.highlight) if (id >= 0 && id < n) mask[id] = 1; }
    for (let i = 0; i < n; i++) {
      const c = clusters[i], v = mask && !mask[i] ? DIM : table[c < 0 ? 0 : c + 1];
      if (w[i] !== v) { w[i] = v; changed = true; }
    }
    return changed;
  }
  /** Uploads the buffers in `mask` (bit i = attribute i); same-size data goes through bufferSubData. */
  upload(mask = 31) {
    if (this.lost || !this.from) return;
    const gl = this.gl, data = [this.from, this.to, this.colorFrom, this.colorTo, this.weights];
    for (let i = 0; i < 5; i++) {
      if (!(mask >> i & 1) || !data[i]) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[i]);
      if (this.uploaded[i] === data[i].length) gl.bufferSubData(gl.ARRAY_BUFFER, 0, data[i]);
      else { gl.bufferData(gl.ARRAY_BUFFER, data[i], gl.DYNAMIC_DRAW); this.uploaded[i] = data[i].length; }
    }
  }
  emphasis() {
    if (this.weigh()) { this.upload(16); this.dirty = true; }
    this.labelsDirty = true; this.wake();
  }
  hideClusters(ids) { this.hiddenClusters=ids; this.emphasis(); this.requestPick(); }
  highlightCells(ids) { if (ids == null && this.highlight === null) return; this.highlight = ids == null ? null : new Set(ids); this.emphasis(); }
  highlightCluster(id) { id ??= null; if (id === this.clusterHighlight) return; this.clusterHighlight = id; this.emphasis(); }
  select(id) { if (id === this.selected) return; this.selected = id; this.dirty = true; this.wake(); }
  /** Go deeper (qc-deeper.js): rings around points (indices: label - 1) by kind, 1 the window's cell (white), 2 a
   *  neighbour its RNA seems to come from (magenta), 3 one that looks like its own kind (grey): [[index, kind]], at most
   *  6; null or [] for none (the map draws exactly as without them). Redraws only when they changed. */
  ringCells(rings) {
    const list = (Array.isArray(rings) ? rings : []).filter(r => Array.isArray(r) && Number.isInteger(r[0]) && r[0] >= 0 && [1, 2, 3].includes(r[1])).slice(0, 6);
    const key = list.map(r => r.join(':')).join(',');
    if (key === (this.ringKey ?? '')) return false;
    this.ringKey = key; this.rings = list; this.dirty = true; this.wake();
    return true;
  }
  setAutoRotate() { this.autoRotate = false; }   // a 2-D map does not rotate
  fit() {
    if (!this.to || !this.width || !this.height) return;
    // Fit the bulk of the cells (2nd-98th percentile per axis): a few stragglers must not shrink the map.
    const b = this.bounds || fitBounds(this.to, this.count);
    if (!b) return;
    const lo = [b.x[0], b.y[0]], hi = [b.x[1], b.y[1]];
    this.center = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, 0];
    const radius = Math.max(.01, Math.hypot((hi[0]-lo[0])/2, (hi[1]-lo[1])/2) * 1.08);
    const halfAngle = Math.atan(Math.tan(Math.PI/8)*Math.min(1,this.width/this.height));
    this.distance = radius / Math.sin(halfAngle) * 1.12; this.fitted = true; this.cameraChanged();
  }
  cameraChanged() { this.cameraVersion++; this.dirty = this.labelsDirty = true; this.wake(); }
  project(point, basis = this.basis()) {
    const d = point.map((v,k) => v-this.center[k]), dot = v => d.reduce((s,x,k) => s+x*v[k],0);
    const z = this.distance-dot(basis.back);
    if (z <= this.distance*.001 || z >= this.distance*100) return null;
    const scale = this.height*1.20710678/z;
    return { x: this.width/2+dot(basis.right)*scale, y:this.height/2-dot(basis.up)*scale, z };
  }
  pan(dx, dy, distance = this.distance) {
    const b = this.basis(), scale = distance/(Math.max(1,this.height)*1.20710678);
    this.center = this.center.map((v,k) => v+(-dx*b.right[k]+dy*b.up[k])*scale); this.cameraChanged();
  }
  point(e) { const r = this.clientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  events() {
    this.listen(this.canvas,'contextmenu',e => e.preventDefault());
    this.listen(this.canvas,'pointerenter',() => { this.rectStale = true; });
    this.listen(this.canvas,'pointerdown',e => {
      if (e.button !== 0 && e.button !== 2) return;
      const p = this.point(e);
      this.drag = { ...p, start:p, pan:e.button===2||e.shiftKey, button:e.button, id:e.pointerId, moved:false };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.listen(this.canvas,'pointermove',e => {
      const p = this.point(e); this.pointer=p;
      const d=this.drag;
      if (!d) { this.requestPick(); return; }   // picked once per frame, not per event
      if (d.id!==e.pointerId) return;
      const dx=p.x-d.x, dy=p.y-d.y;
      d.moved ||= Math.hypot(p.x-d.start.x,p.y-d.start.y)>3;
      if(d.moved) this.userCamera=true;
      if(dx||dy) this.pan(dx,dy);   // 2-D map: dragging pans (no rotation)
      d.x=p.x; d.y=p.y;
    });
    this.listen(this.canvas,'pointerup',e => {
      const d=this.drag; if(!d || d.id!==e.pointerId) return;
      this.drag=null; this.canvas.releasePointerCapture(e.pointerId);
      this.pointer=this.point(e);
      if(!d.moved && d.button===0) { const id=this.pick(); this.select(id); this.onSelect(id); }   // the point as drawn now
      this.requestPick();
    });
    for(const event of ['pointercancel','lostpointercapture']) this.listen(this.canvas,event,() => { this.drag=null; this.requestPick(); });
    this.listen(this.canvas,'pointerleave',() => { this.pointer=null; this.setHover(null); });
    this.listen(this.canvas,'dblclick',() => { this.userCamera=false; this.fit(); this.requestPick(); });
    this.listen(this.canvas,'wheel',e => {
      e.preventDefault(); this.userCamera=true; const p=this.point(e), old=this.distance; this.pointer=p;
      const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?this.height:1);
      this.distance=Math.max(1e-5,Math.min(1e10,old*Math.exp(Math.max(-1,Math.min(1,delta*.001)))));
      this.pan(p.x-this.width/2,p.y-this.height/2,this.distance-old);
      // Zooming keeps the point under the pointer in place: hover is re-picked once the wheel pauses.
      clearTimeout(this.pickTimer); this.pickTimer=setTimeout(() => { this.pickTimer=null; this.requestPick(); },120);
    },{passive:false});
  }
  requestPick() { if (this.pointer && !this.drag) { this.pickPending = true; this.wake(); } }
  setHover(id) { if(id!==this.hover) {this.hover=id;this.onHover(id);this.dirty=true;this.wake();} }
  /** The point under the pointer as drawn at morph progress t: while a morph runs, one scan of the moving points;
   * once it has settled (t = 1, drawn = targets), the cached grid of the target positions. */
  pick(t = this.transition()) {
    if (!this.pointer || !this.count || !this.width || !this.height) return null;
    const view = { center: this.center, distance: this.distance, width: this.width, height: this.height, basis: this.basis() };
    if (t < 1) { this.drawnPicks++; return pickDrawn(this.from, this.to, t, this.count, this.pointer.x, this.pointer.y, view, this.clusters, this.hiddenClusters); }
    let g = this.grid;
    if (!g || g.version !== this.version || g.camera !== this.cameraVersion) {
      g = this.grid = buildPickGrid(this.to, this.count, view, g);
      g.version = this.version; g.camera = this.cameraVersion; this.gridBuilds++;
    }
    return pickNearest(g, this.pointer.x, this.pointer.y, this.clusters, this.hiddenClusters);
  }
  wake() {
    if(this.disposed || this.lost || this.covered || this.raf != null || this.inFrame) return;
    this.raf=requestAnimationFrame(time => {this.raf=null;this.draw(time);});
  }
  /** Compare's full-screen view covers the page (analysis.js setCovered), or no longer does. Covered, nothing is drawn:
   *  the frame asked for goes and none is asked (a flight on its way, a resize, rings); uncovered, what is pending is
   *  drawn (a flight lands where its time says by then). */
  setCovered(on) {
    on = !!on;
    if (on === this.covered) return;
    this.covered = on;
    if (on) { if (this.raf != null) { cancelAnimationFrame(this.raf); this.raf = null; } return; }
    if (this.morphing || this.dirty || this.labelsDirty || this.pickPending || this.sizeStale) this.wake();
  }
  draw(time = performance.now()) {
    if(this.disposed || this.lost) return;
    this.inFrame = true;
    try {
      if (this.sizeStale) this.measure();   // the only layout read, before any write of this frame
      const t = this.transition(time), visible = this.width > 0 && this.height > 0, morph = this.morphing;
      // The morph ends in this frame: other points may now lie under a resting pointer.
      if (morph && t >= 1) { this.morphing = false; this.requestPick(); }
      if (this.pickPending) { this.pickPending = false; if (!this.drag) this.setHover(this.pick(t)); }   // as drawn below
      if (visible && (this.dirty || morph)) this.render(t);
      if (visible && (this.labelsDirty || morph)) this.layoutOverlay(t);
      this.dirty = this.labelsDirty = false;
    } finally { this.inFrame = false; }
    if (this.morphing || this.dirty || this.labelsDirty || this.pickPending || this.sizeStale) this.wake();
  }
  render(t) {
    const b=this.basis(), gl=this.gl, u=this.uniforms;
    this.drawCount++;harness?.count('umapDraws');
    gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(.043,.055,.075,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.program);gl.bindVertexArray(this.vao);
    gl.uniform1f(u.Mix,t);gl.uniform1f(u.Distance,this.distance);gl.uniform1f(u.Aspect,this.width/Math.max(1,this.height));
    gl.uniform1f(u.Ratio,this.ratio);gl.uniform1f(u.Selected,this.selected??-1);gl.uniform1f(u.Hover,this.hover??-1);
    gl.uniform3fv(u.Center,this.center);gl.uniform3fv(u.Right,b.right);gl.uniform3fv(u.Up,b.up);gl.uniform3fv(u.Back,b.back);
    const rings=this.rings||[];gl.uniform1i(u.RingCount,rings.length);
    if(rings.length){const at=new Float32Array(6).fill(-1),kinds=new Float32Array(6);rings.forEach(([i,k],n)=>{at[n]=i;kinds[n]=k;});gl.uniform1fv(u.Rings,at);gl.uniform1fv(u.RingKinds,kinds);}
    if(this.count) gl.drawArrays(gl.POINTS,0,this.count);
  }
  /** Places the labels and writes only the styles that changed (transform, so moving labels cost no layout). */
  layoutOverlay(t) {
    const b=this.basis(), W=this.width, H=this.height;
    if (this.gizmoBasis !== `${this.yaw},${this.pitch}`) {
      this.gizmoBasis = `${this.yaw},${this.pitch}`;
      for(let k=0;k<3;k++){const node=this.axes[k],x=b.right[k]*25,y=-b.up[k]*25;node.style.width=`${Math.max(5,Math.hypot(x,y))}px`;node.style.transform=`rotate(${Math.atan2(y,x)}rad)`;}
    }
    for (const l of this.labels) {
      l.ax = l.ay = NaN;
      if (this.hiddenClusters?.has(l.id)) continue;
      const x=mix(l.from[0],l.target[0],t)-this.center[0],y=mix(l.from[1],l.target[1],t)-this.center[1],z=mix(l.from[2],l.target[2],t)-this.center[2];
      const depth=this.distance-x*b.back[0]-y*b.back[1]-z*b.back[2];
      if(!(depth>this.distance*.001&&depth<this.distance*100))continue;
      const scale=H*1.20710678/depth, px=W/2+(x*b.right[0]+y*b.right[1]+z*b.right[2])*scale, py=H/2-(x*b.up[0]+y*b.up[1]+z*b.up[2])*scale;
      if(px>=0&&py>=0&&px<=W&&py<=H) { l.ax=px; l.ay=py; }
    }
    placeLabels(this.labels, W, H);
    for (const l of this.labels) {
      const s = l.node.style;
      if (l.placed !== l.shown) s.display = (l.shown = l.placed) ? 'block' : 'none';
      if (!l.placed) continue;
      const left = Math.round(l.x), top = Math.round(l.y), opacity = this.clusterHighlight===null||this.clusterHighlight===l.id?'1':'.25';
      if (left !== l.left || top !== l.top) { s.transform = `translate(${left}px,${top}px)`; l.left = left; l.top = top; }
      if (opacity !== l.opacity) s.opacity = l.opacity = opacity;
    }
  }
  dispose() {
    this.disposed=true;cancelAnimationFrame(this.raf);this.raf=null;clearTimeout(this.pickTimer);this.observer.disconnect();this.listeners.forEach(fn=>fn());this.overlay.remove();
    this.canvas.style.touchAction=this.oldTouchAction;
    if(!this.lost) {this.buffers.forEach(b=>this.gl.deleteBuffer(b));this.gl.deleteVertexArray(this.vao);this.gl.deleteProgram(this.program);}
  }
}
