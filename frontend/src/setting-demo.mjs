// The "?" beside the test corner's Setting choice: hovering it (or focusing it, or a tap) opens a popover that plays
// what the chosen setting does to a made-up cell, one panel per value. Each demo is a made-up tissue (nine copies of a
// few cells in one 150 µm square, so edges move at a real square's pace, started from cells only, every other setting
// at proseg's defaults) run through proseg-live once per value; each panel shows the copy nearest the middle of 27. Its
// data (frontend/demos/<key>.json, built from the runs' own where exports and counts) is loaded on first open and kept.
//
// A popover holds one sentence, one panel per value (the value and one number under it), a thin progress line with the
// iteration of the frame shown, and a key. Every dot takes the colour of the cell whose count keeps it (proseg's count,
// as Go deeper reads it; grey: no cell's count); pink marks a neighbour's RNA this cell's count keeps. All panels share
// one clock: a pause at the start, then each captured frame to the next (the outlines move a grid square at a time
// between two frames, in proseg's own 2 µm squares while it settles; dots change colour as the edge passes them, or at
// a fixed spread), a pause at the end, then a fade back to the start. Pink dots breathe as Go deeper's glow does.
// Painting runs only while a popover is open (at most 30 paints a second while things move, 12 while only the glow
// breathes); reduced or calm motion shows each panel's end, still. The popover sits beside the test corner, at the
// "?"'s height, never over the form; where the stage has no room beside the corner, above (else below) the "?".
import { bannedWords } from './qc-copy.mjs';

/** The settings that have a demo: proseg's flag -> the demo's key (its file frontend/demos/<key>.json). */
export const DEMO_KEYS = Object.freeze({ 'cell-compactness': 'compactness', 'prior-weight': 'hold', 'diffusion-probability': 'far-drift',
  'burnin-voxel-size': 'starting-grid', 'ab-nihlo-bubble-prob': 'background-gaps', 'no-diffusion': 'drift', 'diffusion-sigma-near': 'short-drift',
  'density-bins': 'crowding-levels', 'voxel-layers': 'depth-layers' });
/** The words every popover shares (each demo's own come with its data). */
export const DEMO_TEXT = Object.freeze({
  button: '?',
  buttonLabel: 'How this setting works',
  buttonTooltip: 'Hover to see what this setting does to a made-up cell, one panel per value.',
  iteration: 'Iteration {i} of {n} · {phase}',
  phases: Object.freeze(['settling', 'refining', 'final settling']),
  progressTip: "Proseg's schedule: settling to iteration 200, refining to 400, final settling to 450. The panels show proseg's state every 5 to 50 iterations; the motion in between is drawn in. Nine copies share one 150 µm square, so edges move at about a real square's pace.",
  keyCell: 'This cell',
  keyCellTip: "This cell's outline in each run: a drawing. Each dot takes the colour of the cell whose count keeps it.",
  keyPink: 'Neighbour RNA counted',
  keyPinkTip: "Pink: a neighbour's RNA that proseg counts for this cell, a fifth of it or more, as in Go deeper. Brighter: more of it.",
  keyNone: 'Not counted',
  keyNoneTip: "RNA no cell's count keeps: proseg reads it as background.",
  default: "proseg's default",
  infoLabel: 'About this example',
  loading: 'Loading the example…',
  failed: "Couldn't load this example.",
});
/** Every word a popover can show, its own and its demos' (for the words tests). */
export function demoTexts(specs = []) {
  const own = Object.values(DEMO_TEXT).flatMap(v => (Array.isArray(v) ? v : [v]));
  const theirs = specs.flatMap(s => [s.name, s.sentence, s.info, s.statText, s.statTip, s.spotKey, s.spotTip, s.keyNoneTip, s.sideKey, s.sideTip,
    s.progressTip].filter(Boolean));
  return [...own, ...theirs];
}
export { bannedWords };

// ---- the demos' data, decoded once per demo (no DOM here: the node tests load this part alone) ----------------------
// Per captured frame and value: each cell's outline (SVG path data on whole µm: M x y, then h/v steps, Z), the dots whose
// state changed since the previous frame (packed little-endian uint16: dot << 4 | state), this cell's keep per gene
// (x 1000) and the frame's number for the panel. A dot's state is the cell whose count keeps it: 1 this cell, 2-4 its
// neighbours, 5-8 a cell of another copy (its kind + 4), 0 no cell's count keeps it.
export function bytes(b64) {
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
export function u16(b64) {
  const b = bytes(b64), out = new Uint16Array(b.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = b[2 * i] | (b[2 * i + 1] << 8);
  return out;
}
function digits(text) {
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) - 48;
  return out;
}
/** A cell's outline (SVG path data: M, h, v, H, V, Z on whole µm) filled even-odd onto `grid` (W x H, 1 µm squares, a
 *  square's centre inside counts) with `label`; the squares filled. */
export function fillPath(grid, W, H, d, label) {
  const edges = [], re = /([MmHhVvZz])([^MmHhVvZz]*)/g;
  let m, x = 0, y = 0, sx = 0, sy = 0;
  while ((m = re.exec(d))) {
    const c = m[1], nums = (m[2].match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
    if (c === 'M' || c === 'm') {
      if (c === 'M') { x = nums[0]; y = nums[1]; } else { x += nums[0]; y += nums[1]; }
      sx = x; sy = y;
    } else if (c === 'h') { for (const n of nums) x += n; }
    else if (c === 'H') { x = nums[nums.length - 1]; }
    else if (c === 'v' || c === 'V') {
      for (const n of nums) { const y1 = c === 'v' ? y + n : n; edges.push([x, Math.min(y, y1), Math.max(y, y1)]); y = y1; }
    } else {   // Z: back to the loop's start (a vertical closing step counts)
      if (x === sx && y !== sy) edges.push([x, Math.min(y, sy), Math.max(y, sy)]);
      x = sx; y = sy;
    }
  }
  let filled = 0;
  for (let j = 0; j < H; j++) {
    const yc = j + 0.5, xs = [];
    for (const e of edges) if (e[1] < yc && yc < e[2]) xs.push(e[0]);
    if (xs.length < 2) continue;
    xs.sort((a, b) => a - b);
    for (let p = 0; p + 1 < xs.length; p += 2) {
      const a = Math.max(0, Math.round(xs[p])), b = Math.min(W, Math.round(xs[p + 1]));
      for (let i = a; i < b; i++) { grid[j * W + i] = label; filled++; }
    }
  }
  return filled;
}
export function rasterize(paths, W, H) {
  const grid = new Uint8Array(W * H);
  for (let c = 0; c < paths.length; c++) fillPath(grid, W, H, paths[c], c + 1);
  return grid;
}
/** A small fixed hash of an integer to [0, 1). */
export function hash01(n) {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
/** How one frame's grid turns into the next: each square that changes cell, and when in the step it does (0-1). Proseg
 *  moves an edge one grid square at a time, so the squares move as a front: those next to their new cell first, then
 *  outward through the squares changing to that cell, with a little spread; squares with no such path halfway. `block`
 *  is proseg's grid square in µm for this step: 2 while it settles (and from the settling's last frame to the upscale),
 *  when whole 2 µm squares move together, else 1. At 0 the grid is the first frame's and at 1 the next frame's. */
export function planStep(g0, g1, W, H, seed, block) {
  // proseg's grid square in µm for this step: whole squares of it move together (the starting grid's, 2 µm by default,
  // while it settles), or [across, down] when the two differ (the depth layers' side cut: 2 µm across, one layer down)
  const BX = Array.isArray(block) ? (block[0] > 1 ? block[0] : 1) : block > 1 ? block : 1;
  const BY = Array.isArray(block) ? (block[1] > 1 ? block[1] : 1) : BX;
  const BW = Math.ceil(W / BX), BH = Math.ceil(H / BY), nb = BW * BH;
  const bnew = new Int16Array(nb).fill(-1);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const v = j * W + i;
    if (g0[v] !== g1[v]) bnew[((j / BY) | 0) * BW + ((i / BX) | 0)] = g1[v];
  }
  const changed = [];
  for (let v = 0; v < nb; v++) if (bnew[v] >= 0) changed.push(v);
  const count = changed.length, pos = new Int32Array(nb).fill(-1), dist = new Int32Array(count), queue = new Int32Array(count);
  let qn = 0;
  for (let k = 0; k < count; k++) pos[changed[k]] = k;
  const blockLabel0 = b => { const bi = b % BW, bj = (b / BW) | 0; return g0[Math.min(H - 1, bj * BY) * W + Math.min(W - 1, bi * BX)]; };
  const near = (b, f) => {
    const bi = b % BW;
    if (bi > 0) f(b - 1); if (bi < BW - 1) f(b + 1); if (b >= BW) f(b - BW); if (b + BW < nb) f(b + BW);
  };
  for (let k = 0; k < count; k++) {
    const cb = changed[k], L = bnew[cb]; let hit = false;
    near(cb, u => { if (blockLabel0(u) === L) hit = true; });
    if (hit) { dist[k] = 1; queue[qn++] = k; }
  }
  for (let qi = 0; qi < qn; qi++) {
    const kk = queue[qi], vb = changed[kk], LL = bnew[vb];
    near(vb, u => { const ku = pos[u]; if (ku >= 0 && !dist[ku] && bnew[u] === LL) { dist[ku] = dist[kk] + 1; queue[qn++] = ku; } });
  }
  let maxd = 1;
  for (let k = 0; k < count; k++) if (dist[k] > maxd) maxd = dist[k];
  const tb = new Float32Array(count);
  for (let k = 0; k < count; k++) tb[k] = 0.04 + 0.92 * (((dist[k] || Math.ceil(maxd / 2)) - 1 + hash01(changed[k] * 131 + seed)) / maxd);
  const order = [];     // every changed square takes its block's time
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const v = j * W + i;
    if (g0[v] !== g1[v]) order.push([tb[pos[((j / BY) | 0) * BW + ((i / BX) | 0)]], v, g1[v]]);
  }
  order.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const n = order.length, t = new Float32Array(n), idx = new Int32Array(n), lab = new Uint8Array(n), when = new Float32Array(W * H).fill(-1);
  for (let k = 0; k < n; k++) { t[k] = order[k][0]; idx[k] = order[k][1]; lab[k] = order[k][2]; when[idx[k]] = t[k]; }
  return { t, idx, lab, when, count: n, block: BX === BY ? BX : [BX, BY] };
}
/** The grid `p` (0-1) into a step: the first frame's with every change up to p made (out: a reused array). */
export function gridAt(g0, plan, p, out) {
  out.set(g0);
  for (let k = 0; k < plan.count && plan.t[k] <= p; k++) out[plan.idx[k]] = plan.lab[k];
  return out;
}
/** A demo's decoded data: dots (x, y in µm, origin, role, gene), and per value per frame each dot's state, the grid of
 *  outlines, this cell's keep per gene (0-1) and the panel's number; per frame its grid square (2 or 1 µm). */
export function decode(demo) {
  const n = demo.dots.n, xs = u16(demo.dots.x), ys = u16(demo.dots.y), W = demo.box[0], H = demo.box[1];
  const x = new Float32Array(n), y = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = xs[i] / 8; y[i] = ys[i] / 8; }
  const origin = digits(demo.dots.origin), gene = bytes(demo.dots.gene), start = digits(demo.start), role = demo.dots.role;
  const values = demo.v.map(frames => {
    const labels = [], grids = [], keep = [], stat = [], pinkP = [], cuts = [];
    let cur = start, listed = false;
    for (const f of frames) {
      const next = new Uint8Array(cur), a = u16(f.a);
      for (let t = 0; t < a.length; t++) next[a[t] >> 4] = a[t] & 15;
      labels.push(next); cur = next;
      grids.push(rasterize(f.o, W, H));
      const q = u16(f.q), kq = new Float32Array(q.length);
      for (let g = 0; g < q.length; g++) kq[g] = q[g] / 1000;
      keep.push(kq); stat.push(f.n);
      // the frame's pink dots, when the demo lists them (p: uint16 pairs, dot and P x 1000: Go deeper's glow)
      let pk = null;
      if (typeof f.p === 'string') {
        const pp = u16(f.p); pk = new Float32Array(n); listed = true;
        for (let u = 0; u + 1 < pp.length; u += 2) pk[pp[u]] = pp[u + 1] / 1000;
      }
      pinkP.push(pk);
      if (f.c != null) cuts.push(digits(f.c));          // the side cut (depth layers): one digit per column and layer
    }
    return { labels, grids, keep, stat, plans: [], steps: [], pinkP: listed ? pinkP : null, cuts, sidePlans: [] };
  });
  // Each value's grid square per frame (a number for every value, or a list with one per value: the starting grid's).
  const gridOf = demo.v.map((frames, vi) => demo.grids.map(g => (Array.isArray(g) ? g[vi] : g)));
  // The side cut (a demo with depth layers): a row of the copy's tile seen from the side, proseg's own layers along its
  // 1 µm columns x0..x1 (bottom layer first); the dots in that row, each at its depth (0 bottom, 1 top) in each value's
  // run; the made-up cells' true leaning border there.
  let side = null;
  if (demo.side) {
    const sd = demo.side;
    side = { x0: sd.x[0], x1: sd.x[1], W: sd.x[1] - sd.x[0], y0: sd.y[0], y1: sd.y[1], layers: sd.layers.slice(), dots: u16(sd.dots),
      depth: sd.depth.map(b => { const v = u16(b), o = new Float32Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] / 10000; return o; }),
      lean: (sd.lean || []).map(pts => pts.map(q => [q[0], q[1] / 10000])) };
    if (side.x0 % 2) throw new Error('side cut: its first column must start a 2 µm square');
  }
  return { n, x, y, origin, gene, role, W, H, values, gridOf, side };
}
/** The cell a state is drawn as (its colour): another copy's cell as its kind. */
export const colourOf = state => (state >= 5 ? state - 4 : state);
/** The grid square of step k (frame k to k+1) of value vi: B µm when both frames lie on proseg's B µm grid (B > 1). */
export function stepBlock(dec, k, vi = 0) {
  const g = dec.gridOf[vi] ?? dec.gridOf[0];
  return g[k] > 1 && g[k + 1] === g[k] ? g[k] : 1;
}
/** The plan of step k (frame k to k+1) of value vi, made once. */
export function plan(dec, vi, k) {
  const val = dec.values[vi];
  return val.plans[k] || (val.plans[k] = planStep(val.grids[k], val.grids[k + 1], dec.W, dec.H, 7919 * (k + 1) + 104729 * vi, stepBlock(dec, k, vi)));
}
/** The plan of step k of value vi's side cut, made once: its squares (1 µm by one layer; 2 µm by one layer while proseg
 *  settles) move as a front, as the view's do. */
export function sidePlan(dec, vi, k) {
  const val = dec.values[vi];
  return val.sidePlans[k] || (val.sidePlans[k] = planStep(val.cuts[k], val.cuts[k + 1], dec.side.W, dec.side.layers[vi],
    6151 * (k + 1) + 92821 * vi, stepBlock(dec, k, vi) === 2 ? [2, 1] : 1));
}
/** How strongly a dot is pink in a frame (0-1): a neighbour's RNA (by the toy's own record) that this cell's count
 *  keeps (its state is this cell: keep >= 0.2), as strong as that keep. */
export function pinkOf(dec, vi, k, d) {
  const val = dec.values[vi];
  if (val.pinkP) { const p = val.pinkP[k] ? val.pinkP[k][d] : 0; return p > 0 ? Math.max(0.2, Math.min(1, p)) : 0; }   // a listed one
  if (val.labels[k][d] !== 1 || dec.origin[d] < 2) return 0;
  return Math.max(0.2, Math.min(1, val.keep[k][dec.gene[d]]));
}
/** When in a step a dot changes (0-1): with the square under it when that square changes, else at a fixed spread. */
export function dotWhen(dec, pl, d, seed) {
  const v = Math.min(dec.H - 1, Math.floor(dec.y[d])) * dec.W + Math.min(dec.W - 1, Math.floor(dec.x[d]));
  const t = pl.when[v];
  return t >= 0 ? t : 0.15 + 0.7 * hash01(d * 31 + seed);
}
/** How long each step plays (ms): longer where this cell changes more in any value (its squares, and half its dots that
 *  change state), within [min, max]. */
export function stepDurations(dec, min, max, base, per) {
  const K = dec.values[0].labels.length, out = [];
  for (let k = 0; k + 1 < K; k++) {
    let score = 0;
    for (const val of dec.values) {
      const l0 = val.labels[k], l1 = val.labels[k + 1], g0 = val.grids[k], g1 = val.grids[k + 1];
      let s = 0;
      for (let i = 0; i < g0.length; i++) if (g0[i] !== g1[i] && (g0[i] === 1 || g1[i] === 1)) s++;
      for (let i = 0; i < dec.n; i++) if (l0[i] !== l1[i] && (l0[i] === 1 || l1[i] === 1)) s += 0.5;
      score = Math.max(score, s);
    }
    out.push(Math.max(min, Math.min(max, base + per * score)));
  }
  return out;
}

// ---- drawing ---------------------------------------------------------------------------------------------------------
const START_HOLD = 600, END_HOLD = 1900, XFADE = 450;                  // ms
const STEP_MIN = 200, STEP_MAX = 440, STEP_BASE = 150, STEP_PER = 1.6;     // a step's length by how much changes in it
const FPS_MOVE = 30, FPS_BREATH = 12, BREATH_MS = 1600, BREATH_LOW = 0.6;
const DOT_PX = 1.15, BIG_PX = 2.4, CORE_PX = 1.0, HALO_PX = 4.4, CELL_PX = 2, EDGE_PX = 1, DOT_FADE = 0.25, SIDE_DOT_PX = 1.6;
const MARKS = [200, 400];
export const OPEN_DELAY_MS = 110, CLOSE_DELAY_MS = 200, GAP_PX = 8;

const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const lift = (c, t) => c.map(v => Math.round(v + (255 - v) * t));
// Colours: this cell blue, its neighbours amber (A), green (B) and violet (C), no cell grey; the pink and its glow as
// Go deeper's (glow.mjs: #ff4fa3, a white-pink core). The views are dark as the test corner's; light in the figure theme.
const PALETTES = {
  dark: { view: '#0b0e13', cells: [[92, 170, 255], [245, 166, 35], [62, 196, 142], [180, 142, 255]], none: [104, 114, 128], nonePale: [206, 212, 222],
    dotAlpha: 0.9, fill: [0.16, 0.085, 0.085, 0.085], edge: c => rgba(lift(c, 0.35), 0.62),
    cell: '#ffffff', cellGlow: 'rgba(255,255,255,0.65)', haloBlend: 'screen', haloAlpha: 0.66,
    layerLine: 'rgba(255,255,255,0.28)', leanLine: 'rgba(255,255,255,0.8)',
    halo: [[0, 'rgba(255,79,163,1)'], [0.22, 'rgba(255,79,163,0.75)'], [0.5, 'rgba(255,79,163,0.28)'], [1, 'rgba(255,79,163,0)']],
    core: [[0, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,218,236,1)'], [0.75, 'rgba(255,79,163,0.9)'], [1, 'rgba(255,79,163,0)']] },
  light: { view: '#ffffff', cells: [[30, 110, 212], [196, 108, 0], [18, 136, 92], [124, 84, 214]], none: [150, 160, 172], nonePale: [150, 160, 172],
    dotAlpha: 0.85, fill: [0.11, 0.07, 0.07, 0.07], edge: c => rgba(c, 0.72),
    cell: '#151a21', cellGlow: null, haloBlend: 'source-over', haloAlpha: 0.34,
    layerLine: 'rgba(21,26,33,0.3)', leanLine: 'rgba(21,26,33,0.8)',
    halo: [[0, 'rgba(232,48,140,0.85)'], [0.3, 'rgba(232,48,140,0.42)'], [0.65, 'rgba(232,48,140,0.1)'], [1, 'rgba(232,48,140,0)']],
    core: [[0, 'rgba(206,16,106,1)'], [0.55, 'rgba(224,40,127,1)'], [0.8, 'rgba(224,40,127,0.75)'], [1, 'rgba(224,40,127,0)']] },
};
for (const p of Object.values(PALETTES)) {
  const colours = [rgba(p.none, 1), ...p.cells.map(c => rgba(c, 1))];
  p.dotColor = [...colours, ...colours.slice(1), rgba(p.nonePale, 1)];   // 5-8: another copy's cell, as its kind; 9: "not
                                                                         // counted" drawn pale (a demo with nonePale)
  p.fillRGBA = [[0, 0, 0, 0], ...p.cells.map((c, i) => [c[0], c[1], c[2], Math.round(255 * p.fill[i])])];
  p.edgeColor = [null, p.cell, ...p.cells.slice(1).map(p.edge)];
}
const fillWords = (tpl, values) => String(tpl ?? '').replace(/\{(\w+)\}/g, (m, k) => (values[k] != null ? String(values[k]) : m));
const smooth = x => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };
const phaseOf = it => DEMO_TEXT.phases[it < 200 ? 0 : it < 400 ? 1 : 2];

/** One demo's popover: built on first open, painted only while open. */
class Demo {
  constructor(spec, owner) { this.spec = spec; this.o = owner; this.dec = null; this.el = null; this.isOpen = false; this.raf = 0; this.paints = 0; }
  get doc() { return this.o.doc; }
  get win() { return this.o.win; }
  make(tag, cls, text) { const n = this.doc.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  canvas() { return this.doc.createElement('canvas'); }
  ready() {
    if (this.dec) return;
    this.dec = decode(this.spec);
    this.K = this.dec.values[0].labels.length;
    this.last = this.spec.iters[this.K - 1];
    this.durs = stepDurations(this.dec, STEP_MIN, STEP_MAX, STEP_BASE, STEP_PER);
    let at = START_HOLD; this.cum = [];
    for (const d of this.durs) { this.cum.push(at); at += d; }
    this.stepsEnd = at; this.total = at + END_HOLD + XFADE;
    // The dots a demo points at (far drift: the RNA lying over the border from its own cell): bigger, the rest dimmer.
    const spot = this.spec.spot || '', roles = this.dec.role; this.spot = new Uint8Array(this.dec.n);
    for (let d = 0; d < this.dec.n; d++) if (spot && spot.indexOf(roles[d]) >= 0) this.spot[d] = 1;
    this.dim = spot ? (this.spec.dim || 0.5) : 1;
    // A demo with nonePale draws the RNA no cell's count keeps pale in the dark theme (colour 9), so what a cell's count
    // sets aside stands out on its blue (crowding levels).
    this.none = this.spec.nonePale ? 9 : 0;
  }
  /** The frame reduced motion shows: the demo's own still frame when it names one (the starting grid's: a settling
   *  frame, where its blocks show), else the end. */
  stillFrame() { const f = this.spec.still?.frame; return Number.isInteger(f) && f >= 0 && f < this.K ? f : this.K - 1; }
  build() {
    if (this.el) return;
    const s = this.spec, el = this.make('div', 'sdemo-pop');
    el.id = `sdemo-pop-${s.key}`; el.hidden = true; el.setAttribute('role', 'tooltip'); el.dataset.demo = s.key;
    el.dataset.qcCopy = '';      // the harness's copy check reads its words
    const text = this.make('p', 'sdemo-text', s.sentence);
    const panels = this.make('div', 'sdemo-panels'); panels.style.gap = `${s.gap}px`;
    this.panels = s.values.map((v, vi) => {
      const fig = this.make('figure', 'sdemo-panel'), frame = this.make('div', 'sdemo-frame'), canvas = this.canvas();
      canvas.style.width = `${s.panel[0]}px`; canvas.style.height = `${s.panel[1]}px`;
      canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${s.name} ${v}`);
      frame.style.width = `${s.panel[0]}px`; frame.append(canvas); fig.append(frame);
      const cap = this.make('figcaption', '', s.unit ? `${v} ${s.unit}` : v);
      if (v === s.default) cap.append(this.make('span', 'sdemo-default', ` · ${DEMO_TEXT.default}`));
      const stat = this.make('span', 'sdemo-stat'); stat.title = s.statTip || '';
      cap.append(stat);
      // depth layers: the side cut under the view, as wide, its own canvas
      let side = null;
      if (s.side) {
        const sframe = this.make('div', 'sdemo-frame sdemo-side'), scanvas = this.canvas();
        scanvas.style.width = `${s.panel[0]}px`; scanvas.style.height = `${s.side.h}px`;
        scanvas.setAttribute('role', 'img'); scanvas.setAttribute('aria-label', `${s.name} ${v} · ${s.sideKey || ''}`);
        sframe.style.width = `${s.panel[0]}px`; sframe.append(scanvas); fig.append(sframe);
        side = { canvas: scanvas, ctx: scanvas.getContext('2d'), w: s.panel[0], h: s.side.h, dpr: 0, cw: 0, ch: 0 };
      }
      fig.style.width = `${s.panel[0]}px`; fig.append(cap); panels.append(fig);
      return { vi, canvas, ctx: canvas.getContext('2d'), w: s.panel[0], h: s.panel[1], dpr: 0, cw: 0, ch: 0, stables: [], statNode: stat, shownStat: null, side };
    });
    const prog = this.make('div', 'sdemo-progress'), iter = this.make('span', 'sdemo-iter'), track = this.make('span', 'sdemo-track');
    const bar = this.make('i', 'sdemo-fill');
    prog.title = s.progressTip || DEMO_TEXT.progressTip; track.setAttribute('aria-hidden', 'true'); track.append(bar);
    for (const m of MARKS) { const i = this.make('i', 'sdemo-mark'); i.style.left = `${(100 * m / this.last).toFixed(2)}%`; track.append(i); }
    prog.append(iter, track);
    const key = this.make('div', 'sdemo-key');
    const item = (cls, label, tip) => { const k = this.make('span', cls); k.title = tip || ''; k.append(this.make('i'), this.doc.createTextNode(label)); return k; };
    key.append(item('sdemo-key-cell', DEMO_TEXT.keyCell, DEMO_TEXT.keyCellTip), item('sdemo-key-pink', DEMO_TEXT.keyPink, DEMO_TEXT.keyPinkTip),
      item('sdemo-key-none', DEMO_TEXT.keyNone, s.keyNoneTip || DEMO_TEXT.keyNoneTip));
    if (s.spotKey) key.append(item('sdemo-key-spot', s.spotKey, s.spotTip));
    if (s.sideKey) key.append(item('sdemo-key-side', s.sideKey, s.sideTip));
    if (s.nonePale) el.dataset.none = 'pale';
    const info = this.make('span', 'sdemo-info', 'ⓘ'); info.title = s.info; info.setAttribute('role', 'img'); info.setAttribute('aria-label', DEMO_TEXT.infoLabel);
    key.append(info);
    const sr = this.make('span', 'sdemo-sr', s.info);
    el.append(text, panels, prog, key, sr);
    this.doc.body.append(el);
    el.addEventListener('pointerenter', () => this.o.overPopover(this, true));
    el.addEventListener('pointerleave', () => this.o.overPopover(this, false));
    this.el = el; this.iterNode = iter; this.barNode = bar; this.shownIter = null;
  }
  /** Canvases at the screen's pixel ratio (their drawings again when it changed). */
  size() {
    const dpr = Math.max(1, Math.min(3, this.win.devicePixelRatio || 1));
    for (const pn of this.panels) {
      const cw = Math.round(pn.w * dpr), ch = Math.round(pn.h * dpr);
      if (pn.cw !== cw || pn.ch !== ch) { pn.canvas.width = cw; pn.canvas.height = ch; pn.cw = cw; pn.ch = ch; pn.stables = []; pn.scratch = null; pn.lines = null; }
      pn.dpr = dpr;
      const sp = pn.side;
      if (sp) {
        const sw = Math.round(sp.w * dpr), sh = Math.round(sp.h * dpr);
        if (sp.cw !== sw || sp.ch !== sh) { sp.canvas.width = sw; sp.canvas.height = sh; sp.cw = sw; sp.ch = sh; sp.scratch = null; sp.geom = null; }
        sp.dpr = dpr;
      }
    }
  }
  /** Each grid line's place on a panel (device px, whole pixels: the outlines land on pixels). */
  lines(pn) {
    if (pn.lines) return pn.lines;
    const c = this.spec.crop, W = this.dec.W, H = this.dec.H, kx = pn.cw / (c[2] - c[0]), ky = pn.ch / (c[3] - c[1]);
    const hx = new Float32Array(W + 1), vy = new Float32Array(H + 1);
    for (let i = 0; i <= W; i++) hx[i] = Math.round((i - c[0]) * kx);
    for (let j = 0; j <= H; j++) vy[j] = Math.round((j - c[1]) * ky);
    return (pn.lines = { kx, ky, x0: c[0], y0: c[1], hx, vy });
  }
  /** Step k of value vi (frame k to k+1): the dots that keep their cell and are not pink (by cell), the others with when
   *  each changes (-1: the same cell, pink in either frame). */
  step(vi, k) {
    const val = this.dec.values[vi];
    if (val.steps[k]) return val.steps[k];
    const dec = this.dec, pl = plan(dec, vi, k), l0 = val.labels[k], l1 = val.labels[k + 1];
    const dyn = [], when = [], stable = [[], [], [], [], [], [], [], [], []];
    for (let d = 0; d < dec.n; d++) {
      const a = l0[d], b = l1[d];
      if (a !== b) { dyn.push(d); when.push(dotWhen(dec, pl, d, 977 * (k + 1) + 31 * vi)); }
      else if (pinkOf(dec, vi, k, d) || pinkOf(dec, vi, k + 1, d)) { dyn.push(d); when.push(-1); }
      else stable[a].push(d);
    }
    const whenOf = new Float32Array(dec.n).fill(-2);        // -2: keeps its cell, not pink; else its `when`
    for (let w = 0; w < dyn.length; w++) whenOf[dyn[w]] = when[w];
    return (val.steps[k] = { plan: pl, dyn, when, stable, whenOf });
  }
  sprite(theme, kind) {
    const key = theme + kind, sprites = this.o.sprites;
    if (sprites[key]) return sprites[key];
    const c = this.canvas(), ctx = c.getContext('2d'); c.width = c.height = 64;
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    for (const s of PALETTES[theme][kind]) grad.addColorStop(s[0], s[1]);
    ctx.fillStyle = grad; ctx.fillRect(0, 0, 64, 64);
    return (sprites[key] = c);
  }
  /** A dot `px` device pixels across in a cell's colour (label; 0 no cell): a round sprite, else a whole-pixel square. */
  dot(ctx, theme, label, X, Y, px, round) {
    if (round) {
      const key = `${theme}${label}|${px}`, sprites = this.o.sprites;
      let sp = sprites[key];
      if (!sp) {
        const size = Math.ceil(px) + 2; sp = this.canvas(); sp.width = sp.height = size;
        const c = sp.getContext('2d'); c.fillStyle = PALETTES[theme].dotColor[label]; c.beginPath(); c.arc(size / 2, size / 2, px / 2, 0, 2 * Math.PI); c.fill();
        sprites[key] = sp;
      }
      const h = sp.width / 2; ctx.drawImage(sp, Math.round(X - h), Math.round(Y - h)); return;
    }
    const sz = Math.max(1, Math.round(px)); ctx.fillStyle = PALETTES[theme].dotColor[label];
    ctx.fillRect(Math.round(X - sz / 2), Math.round(Y - sz / 2), sz, sz);
  }
  /** The dots that keep their cell through step k, drawn once (kept for the last three steps drawn). */
  stableLayer(pn, k, theme) {
    const key = `${k}|${theme}`;
    for (const s of pn.stables) if (s.key === key) return s.canvas;
    const entry = pn.stables.length >= 3 ? pn.stables.shift() : { canvas: this.canvas() };
    const cv = entry.canvas; cv.width = pn.cw; cv.height = pn.ch;
    const ctx = cv.getContext('2d'), g = this.lines(pn), st = this.step(pn.vi, k), dec = this.dec, pal = PALETTES[theme];
    const round = pn.dpr >= 1.5, small = DOT_PX * pn.dpr, big = BIG_PX * pn.dpr;
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (let L = 0; L <= 8; L++) {
      for (const d of st.stable[L]) {
        const on = this.spot[d];
        ctx.globalAlpha = pal.dotAlpha * (on ? 1 : this.dim);
        this.dot(ctx, theme, L || this.none, (dec.x[d] - g.x0) * g.kx, (dec.y[d] - g.y0) * g.ky, on ? big : small, round || on);
      }
    }
    ctx.globalAlpha = 1;
    entry.key = key; pn.stables.push(entry);
    return cv;
  }
  /** One panel's picture `p` (0-1) into step k on `ctx`: the view, the cells faintly filled, the dots, the outlines (the
   *  other cells thin in their colours, this cell bold), the pink dots glowing at `level` (the breath). */
  scene(pn, ctx, k, p, level) {
    const theme = this.o.theme(), pal = PALETTES[theme], dec = this.dec, vi = pn.vi, val = dec.values[vi], g = this.lines(pn);
    const st = this.step(vi, k), W = dec.W, H = dec.H;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = pal.view; ctx.fillRect(0, 0, pn.cw, pn.ch);
    const grid = gridAt(val.grids[k], st.plan, p, pn.grid || (pn.grid = new Uint8Array(W * H)));
    if (!pn.fillCanvas) {     // the cells, faintly filled (a W x H image scaled up, each square sharp)
      pn.fillCanvas = this.canvas(); pn.fillCanvas.width = W; pn.fillCanvas.height = H;
      pn.fillCtx = pn.fillCanvas.getContext('2d'); pn.fillImage = pn.fillCtx.createImageData(W, H);
    }
    const px = pn.fillImage.data, cols = pal.fillRGBA;
    for (let v = 0, o = 0; v < W * H; v++, o += 4) { const c = cols[grid[v]]; px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = c[3]; }
    pn.fillCtx.putImageData(pn.fillImage, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(pn.fillCanvas, 0, 0, W, H, g.hx[0], g.vy[0], g.hx[W] - g.hx[0], g.vy[H] - g.vy[0]);
    // The dots that keep their cell, then those that change (each fading from its colour to the next as the edge passes
    // it) and the pink ones (gathered for the glow).
    ctx.drawImage(this.stableLayer(pn, k, theme), 0, 0);
    const round = pn.dpr >= 1.5, small = DOT_PX * pn.dpr, big = BIG_PX * pn.dpr, glow = [], l0 = val.labels[k], l1 = val.labels[k + 1];
    for (let i = 0; i < st.dyn.length; i++) {
      // A dot fades over DOT_FADE of the step around its time, kept inside the step: each frame exactly at 0 and at 1.
      const d = st.dyn[i], t = st.when[i], s = t < 0 ? p : smooth((p - Math.max(DOT_FADE / 2, Math.min(1 - DOT_FADE / 2, t))) / DOT_FADE + 0.5);
      const pk0 = pinkOf(dec, vi, k, d), pk1 = pinkOf(dec, vi, k + 1, d);
      const X = (dec.x[d] - g.x0) * g.kx, Y = (dec.y[d] - g.y0) * g.ky, a = l0[d] || this.none, b = l1[d] || this.none;
      const on = this.spot[d], dpx = on ? big : small, al = pal.dotAlpha * (on ? 1 : this.dim), rd = round || on;
      if (a === b && !pk0 && !pk1) { ctx.globalAlpha = al; this.dot(ctx, theme, a, X, Y, dpx, rd); }
      else {
        if (!pk0 && s < 1) { ctx.globalAlpha = al * (1 - s); this.dot(ctx, theme, a, X, Y, dpx, rd); }
        if (!pk1 && s > 0) { ctx.globalAlpha = al * s; this.dot(ctx, theme, b, X, Y, dpx, rd); }
      }
      const pink = (1 - s) * pk0 + s * pk1;
      if (pink > 0.02) glow.push(X, Y, pink);
    }
    ctx.globalAlpha = 1;
    this.outlines(ctx, grid, g, pal, pn.dpr);
    this.glow(ctx, glow, pal, theme, level, pn.dpr);
    if (dec.side) this.cutMark(ctx, g, pal, pn);
  }
  /** Where the side cut lies, on the view: a fine dotted line across it with a short tick at each end. */
  cutMark(ctx, g, pal, pn) {
    const sd = this.dec.side, Y = Math.round(((sd.y0 + sd.y1) / 2 - g.y0) * g.ky) + 0.5, t = Math.round(5 * pn.dpr), lw = Math.max(1, Math.round(pn.dpr));
    ctx.save(); ctx.strokeStyle = pal.cell; ctx.lineWidth = lw;
    ctx.globalAlpha = 0.42; ctx.setLineDash([lw, 3 * lw]); ctx.beginPath(); ctx.moveTo(t, Y); ctx.lineTo(pn.cw - t, Y); ctx.stroke();
    ctx.globalAlpha = 0.85; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(0, Y); ctx.lineTo(t, Y); ctx.moveTo(pn.cw - t, Y); ctx.lineTo(pn.cw, Y); ctx.stroke();
    ctx.restore();
  }
  /** The side cut's picture `p` (0-1) into step k: proseg's layers along the cut, bottom layer at the bottom, each square
   *  faintly filled in its cell's colour, a fine line between layers, the made-up cells' true leaning border dashed, the
   *  outlines as the view's (this cell bold); the dots of the cut's row at their depth, coloured as in the view at the
   *  same moment, the pink ones glowing. */
  sideScene(pn, ctx, k, p, level) {
    const theme = this.o.theme(), pal = PALETTES[theme], dec = this.dec, sd = dec.side, vi = pn.vi, val = dec.values[vi], sp = pn.side;
    const N = sd.layers[vi], W = sd.W, c = this.spec.crop;
    if (!sp.geom) {
      const kx = sp.cw / (c[2] - c[0]), hx = new Float32Array(W + 1), vy = new Float32Array(N + 1);
      for (let i = 0; i <= W; i++) hx[i] = Math.round((sd.x0 + i - c[0]) * kx);
      for (let j = 0; j <= N; j++) vy[j] = Math.round(j * sp.ch / N);       // row j of the flipped grid: layer N - 1 - j
      sp.geom = { kx, x0: c[0], hx, vy, flip: new Uint8Array(W * N) };
    }
    const g = sp.geom;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = pal.view; ctx.fillRect(0, 0, sp.cw, sp.ch);
    const cut = gridAt(val.cuts[k], sidePlan(dec, vi, k), p, sp.grid || (sp.grid = new Uint8Array(W * N)));
    for (let j = 0; j < N; j++) for (let i = 0; i < W; i++) { const cv = cut[j * W + i]; g.flip[(N - 1 - j) * W + i] = cv >= 5 ? 0 : cv; }
    const cols = pal.fillRGBA;
    for (let j = 0; j < N; j++) for (let i = 0; i < W; i++) {
      const L = g.flip[j * W + i]; if (!L) continue;
      const f = cols[colourOf(L)];
      ctx.fillStyle = `rgba(${f[0]},${f[1]},${f[2]},${(f[3] / 255).toFixed(3)})`;
      ctx.fillRect(g.hx[i], g.vy[j], g.hx[i + 1] - g.hx[i], g.vy[j + 1] - g.vy[j]);
    }
    ctx.fillStyle = pal.layerLine;
    for (let j = 1; j < N; j++) ctx.fillRect(0, g.vy[j], sp.cw, Math.max(1, Math.round(sp.dpr / 2)));
    const lean = sd.lean[vi];      // the made-up cells' true border along the cut (it leans): dashed
    if (lean && lean.length > 1) {
      const lw = Math.max(1, Math.round(sp.dpr)), off = lw % 2 ? 0.5 : 0;
      ctx.save(); ctx.strokeStyle = pal.leanLine; ctx.lineWidth = lw; ctx.setLineDash([3 * lw, 2 * lw]); ctx.lineCap = 'butt';
      ctx.beginPath();
      lean.forEach((q, n) => { const X = Math.round((q[0] - g.x0) * g.kx) + off, Y = (1 - q[1]) * sp.ch; if (n) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
      ctx.stroke(); ctx.restore();
    }
    // the dots, as in the view: each fades to its next colour at its moment in the step (the view's), pink ones glow
    const st = this.step(vi, k), l0 = val.labels[k], l1 = val.labels[k + 1], depth = sd.depth[vi], glow = [];
    const round = sp.dpr >= 1.5, dpx = SIDE_DOT_PX * sp.dpr, al = pal.dotAlpha;
    for (let n = 0; n < sd.dots.length; n++) {
      const d = sd.dots[n], t = st.whenOf[d], X = (dec.x[d] - g.x0) * g.kx, Y = (1 - depth[n]) * sp.ch, a = l0[d] || this.none, b = l1[d] || this.none;
      const s = t === -2 ? 0 : t < 0 ? p : smooth((p - Math.max(DOT_FADE / 2, Math.min(1 - DOT_FADE / 2, t))) / DOT_FADE + 0.5);
      const pk0 = pinkOf(dec, vi, k, d), pk1 = pinkOf(dec, vi, k + 1, d);
      if (a === b && !pk0 && !pk1) { ctx.globalAlpha = al; this.dot(ctx, theme, a, X, Y, dpx, round); }
      else {
        if (!pk0 && s < 1) { ctx.globalAlpha = al * (1 - s); this.dot(ctx, theme, a, X, Y, dpx, round); }
        if (!pk1 && s > 0) { ctx.globalAlpha = al * s; this.dot(ctx, theme, b, X, Y, dpx, round); }
      }
      const pink = (1 - s) * pk0 + s * pk1;
      if (pink > 0.02) glow.push(X, Y, pink);
    }
    ctx.globalAlpha = 1;
    this.outlines(ctx, g.flip, g, pal, sp.dpr, W, N);
    this.glow(ctx, glow, pal, theme, level, sp.dpr);
  }
  /** Each cell's outline on its own side of the line (as the app's views draw them): runs of square edges between two
   *  cells, merged along rows and columns, whole device pixels. */
  outlines(ctx, grid, g, pal, dpr, gw = 0, gh = 0) {
    const W = gw || this.dec.W, H = gh || this.dec.H, hx = g.hx, vy = g.vy;
    const wid = [0, Math.max(1, Math.round(CELL_PX * dpr)), 0, 0, 0];
    for (let L = 2; L <= 4; L++) wid[L] = Math.max(1, Math.round(EDGE_PX * dpr));
    const half = wid.map(w => w / 2), paths = [null, new Path2D(), new Path2D(), new Path2D(), new Path2D()];
    const seg = (L, x0, y0, x1, y1) => { paths[L].moveTo(x0, y0); paths[L].lineTo(x1, y1); };
    for (let j = 0; j < H; j++) {
      const row = j * W; let tL = 0, tS = 0, bL = 0, bS = 0;
      for (let i = 0; i <= W; i++) {
        const L = i < W ? grid[row + i] : 0;
        const tp = L && (j === 0 || grid[row - W + i] !== L) ? L : 0, bt = L && (j === H - 1 || grid[row + W + i] !== L) ? L : 0;
        if (tp !== tL) { if (tL) seg(tL, hx[tS], vy[j] + half[tL], hx[i], vy[j] + half[tL]); tL = tp; tS = i; }
        if (bt !== bL) { if (bL) seg(bL, hx[bS], vy[j + 1] - half[bL], hx[i], vy[j + 1] - half[bL]); bL = bt; bS = i; }
      }
    }
    for (let i = 0; i < W; i++) {
      let lL = 0, lS = 0, rL = 0, rS = 0;
      for (let j = 0; j <= H; j++) {
        const L = j < H ? grid[j * W + i] : 0;
        const lf = L && (i === 0 || grid[j * W + i - 1] !== L) ? L : 0, rt = L && (i === W - 1 || grid[j * W + i + 1] !== L) ? L : 0;
        if (lf !== lL) { if (lL) seg(lL, hx[i] + half[lL], vy[lS], hx[i] + half[lL], vy[j]); lL = lf; lS = j; }
        if (rt !== rL) { if (rL) seg(rL, hx[i + 1] - half[rL], vy[rS], hx[i + 1] - half[rL], vy[j]); rL = rt; rS = j; }
      }
    }
    ctx.lineCap = 'butt'; ctx.globalAlpha = 1;
    for (let L = 2; L <= 4; L++) { ctx.lineWidth = wid[L]; ctx.strokeStyle = pal.edgeColor[L]; ctx.stroke(paths[L]); }
    if (pal.cellGlow) {   // this cell bold, with a soft glow in the dark theme: two wide faint strokes under it
      ctx.strokeStyle = pal.cellGlow;
      ctx.globalAlpha = 0.18; ctx.lineWidth = wid[1] + 4 * dpr; ctx.stroke(paths[1]);
      ctx.globalAlpha = 0.3; ctx.lineWidth = wid[1] + 2 * dpr; ctx.stroke(paths[1]);
      ctx.globalAlpha = 1;
    }
    ctx.lineWidth = wid[1]; ctx.strokeStyle = pal.cell; ctx.stroke(paths[1]);
  }
  /** The pink dots: a soft halo each (as strong as its pink times the breath; softer in a crowd, so a crowd deepens to
   *  pink rather than burning out to white) and a bright core. */
  glow(ctx, glow, pal, theme, level, dpr) {
    const n = glow.length / 3; if (!n) return;
    let lit = 0; for (let i = 0; i < n; i++) if (glow[3 * i + 2] > 0.1) lit++;
    const crowd = Math.max(0.35, Math.min(1, Math.sqrt(24 / Math.max(1, lit)))), rh = HALO_PX * dpr, rc = 1.7 * CORE_PX * dpr;
    const halo = this.sprite(theme, 'halo'), core = this.sprite(theme, 'core');
    ctx.save();
    ctx.globalCompositeOperation = pal.haloBlend;
    for (let i = 0; i < n; i++) {
      ctx.globalAlpha = Math.min(1, pal.haloAlpha * glow[3 * i + 2] * level * (0.4 + 0.6 * crowd));
      ctx.drawImage(halo, glow[3 * i] - rh, glow[3 * i + 1] - rh, 2 * rh, 2 * rh);
    }
    ctx.globalCompositeOperation = 'source-over';
    for (let i = 0; i < n; i++) {
      ctx.globalAlpha = Math.min(1, glow[3 * i + 2] * (0.8 + 0.2 * crowd));
      ctx.drawImage(core, glow[3 * i] - rc, glow[3 * i + 1] - rc, 2 * rc, 2 * rc);
    }
    ctx.restore();
  }
  /** Where the loop is `tau` ms in: the step and how far into it, the iteration, and the fade back to the start (x). */
  stateAt(tau) {
    const K = this.K, it = this.spec.iters;
    let t = tau % this.total;
    if (t < START_HOLD) return { k: 0, p: 0, iter: it[0], f: 0, moving: false, x: 0 };
    for (let k = 0; k < this.durs.length; k++) {
      const c0 = this.cum[k];
      if (t < c0 + this.durs[k]) { const p = (t - c0) / this.durs[k]; return { k, p, iter: it[k] + (it[k + 1] - it[k]) * p, f: k, moving: true, x: 0 }; }
    }
    t -= this.stepsEnd;
    if (t < END_HOLD) return { k: K - 2, p: 1, iter: it[K - 1], f: K - 1, moving: false, x: 0 };
    const x = smooth((t - END_HOLD) / XFADE);
    return { k: K - 2, p: 1, iter: it[K - 1] * (1 - x), f: x < 0.5 ? K - 1 : 0, moving: true, x };
  }
  /** Frame f, still (the end: the last frame). */
  frameState(f) {
    const K = this.K;
    if (f >= K - 1) return { k: K - 2, p: 1, iter: this.spec.iters[K - 1], f: K - 1, moving: false, x: 0 };
    return { k: f, p: 0, iter: this.spec.iters[f], f, moving: false, x: 0 };
  }
  paint(st, level) {
    for (const pn of this.panels) {
      this.scene(pn, pn.ctx, st.k, st.p, level);
      if (st.x > 0) {
        if (!pn.scratch) { pn.scratch = this.canvas(); pn.scratch.width = pn.cw; pn.scratch.height = pn.ch; }
        this.scene(pn, pn.scratch.getContext('2d'), 0, 0, level);
        pn.ctx.globalAlpha = st.x; pn.ctx.drawImage(pn.scratch, 0, 0); pn.ctx.globalAlpha = 1;
      }
      const sp = pn.side;
      if (sp) {
        this.sideScene(pn, sp.ctx, st.k, st.p, level);
        if (st.x > 0) {
          if (!sp.scratch) { sp.scratch = this.canvas(); sp.scratch.width = sp.cw; sp.scratch.height = sp.ch; }
          this.sideScene(pn, sp.scratch.getContext('2d'), 0, 0, level);
          sp.ctx.globalAlpha = st.x; sp.ctx.drawImage(sp.scratch, 0, 0); sp.ctx.globalAlpha = 1;
        }
      }
    }
    // The words name the frame shown: the last captured frame reached (a step's own frame until the next is drawn).
    const shown = this.spec.iters[st.f];
    this.barNode.style.width = `${(100 * st.iter / this.last).toFixed(2)}%`;
    if (shown !== this.shownIter) { this.shownIter = shown; this.iterNode.textContent = fillWords(DEMO_TEXT.iteration, { i: shown, n: this.last, phase: phaseOf(shown) }); }
    for (const pn of this.panels) {       // one number a panel, or two ([n, m]: "{n} pink · {m} own")
      const sv = this.dec.values[pn.vi].stat[st.f], two = Array.isArray(sv), n = Math.round(two ? sv[0] : sv), m = two ? Math.round(sv[1]) : null;
      const shown = `${n}|${m}`;
      if (shown !== pn.shownStat) { pn.shownStat = shown; pn.statNode.textContent = fillWords(this.spec.statText, { n, m }); }
    }
    this.paints++;
  }
  breath(tau) { return BREATH_LOW + (1 - BREATH_LOW) * (0.5 - 0.5 * Math.cos(2 * Math.PI * tau / BREATH_MS)); }
  run() {
    this.win.cancelAnimationFrame(this.raf);
    if (this.o.still()) { this.paint(this.frameState(this.stillFrame()), 1); return; }
    const t0 = this.win.performance.now();
    let last = -1e9;
    const tick = now => {
      if (!this.isOpen) return;
      this.raf = this.win.requestAnimationFrame(tick);
      const tau = now - t0, st = this.stateAt(tau), fps = st.moving ? FPS_MOVE : FPS_BREATH;
      if (now - last < 1000 / fps - 3 || this.doc.hidden) return;
      last = now; this.paint(st, this.breath(tau));
    };
    this.paint(this.stateAt(0), this.breath(0));
    this.raf = this.win.requestAnimationFrame(tick);
  }
  show() { this.ready(); this.build(); this.size(); this.el.hidden = false; this.isOpen = true; }
  hide() { this.isOpen = false; this.win.cancelAnimationFrame(this.raf); if (this.el) this.el.hidden = true; }
  remove() { this.hide(); this.el?.remove(); this.el = null; }
}

/** The popover's place (CSS px, fixed): beside the test corner (`corner`'s box) at the "?"'s height, to its right, else
 *  its left; else above the "?", else below it; always inside the window (`vw` x `vh`). `button`, `corner`: boxes
 *  {left, top, right, bottom}; `w`, `h`: the popover's size. */
export function placePopover(button, corner, w, h, vw, vh, gap = GAP_PX) {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const box = corner ?? button, mid = (button.top + button.bottom) / 2;
  const top = clamp(mid - h / 2, gap, Math.max(gap, vh - h - gap));
  if (box.right + gap + w <= vw - gap) return { left: box.right + gap, top, side: 'right' };
  if (box.left - gap - w >= gap) return { left: box.left - gap - w, top, side: 'left' };
  const left = clamp(button.left - 22, gap, Math.max(gap, vw - w - gap));
  if (button.top - gap - h >= gap) return { left, top: button.top - gap - h, side: 'above' };
  return { left, top: Math.min(button.bottom + gap, Math.max(gap, vh - h - gap)), side: 'below' };
}

/** The "?" `button` (beside the Setting choice) wired to its demos. `flag`: the setting chosen now; `still()`: reduced or
 *  calm motion (each panel's end, still); `load(key)`: the demo's data (default: fetch frontend/demos/<key>.json);
 *  `corner()`: the element the popover goes beside (default: the "?"'s test corner); `hook`: the harness (counts
 *  sdemoOpens and sdemoPaints). Returns {setFlag, open, close, isOpen, dispose}. A setting without a demo hides the "?". */
export function attachSettingDemo(button, options = {}) {
  const doc = button.ownerDocument ?? globalThis.document, win = doc.defaultView ?? globalThis;
  const listeners = [], cache = new Map();
  const on = (node, type, fn, opts) => {
    if (typeof node?.addEventListener !== 'function') return;      // a page without a window (the node tests)
    node.addEventListener(type, fn, opts); listeners.push([node, type, fn, opts]);
  };
  const loadJson = options.load ?? (key => fetch(new URL(`../demos/${key}.json`, import.meta.url)).then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }));
  const owner = {
    doc, win, sprites: {},
    theme: () => (doc.documentElement?.dataset?.theme === 'figure' ? 'light' : 'dark'),
    still: () => { try { return typeof options.still === 'function' ? !!options.still() : !!win.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches; } catch { return false; } },
    overPopover: (demo, inside) => { overPop = inside && current === demo; if (!inside) closeSoon(); },
  };
  let flag = options.flag ?? null, current = null, pinned = false, overButton = false, overPop = false, openTimer = 0, closeTimer = 0;
  let wanted = 0;      // the open asked for last (a demo that loads after another was asked for is not shown)
  button.type = 'button'; button.textContent = DEMO_TEXT.button;
  button.setAttribute('aria-label', DEMO_TEXT.buttonLabel); button.title = DEMO_TEXT.buttonTooltip;
  button.setAttribute('aria-expanded', 'false'); button.dataset.sdemo = '';
  const keyOf = f => DEMO_KEYS[f] ?? null;
  const cornerBox = () => (options.corner ? options.corner() : button.closest?.('.qc-tc'))?.getBoundingClientRect?.() ?? null;
  function place(demo) {
    const r = button.getBoundingClientRect(), el = demo.el;
    const p = placePopover(r, cornerBox(), el.offsetWidth, el.offsetHeight, doc.documentElement.clientWidth, win.innerHeight);
    el.style.left = `${Math.round(p.left)}px`; el.style.top = `${Math.round(p.top)}px`; el.dataset.side = p.side;
  }
  function spec(key) {
    if (!cache.has(key)) cache.set(key, Promise.resolve().then(() => loadJson(key)).then(s => new Demo(s, owner)));
    return cache.get(key);
  }
  function open() {
    const key = keyOf(flag);
    if (!key) return Promise.resolve(null);
    win.clearTimeout(closeTimer); win.clearTimeout(openTimer);
    const ask = ++wanted;
    button.dataset.sdemo = 'loading';
    return spec(key).then(demo => {
      if (ask !== wanted || keyOf(flag) !== key) return null;
      if (current && current !== demo) current.hide();
      if (current !== demo || !demo.isOpen) {
        current = demo; demo.show(); place(demo); demo.run();
        button.setAttribute('aria-expanded', 'true'); button.setAttribute('aria-controls', demo.el.id);
        options.hook?.count?.('sdemoOpens');
      }
      button.dataset.sdemo = 'open';
      return demo;
    }, err => {
      if (ask === wanted) { button.dataset.sdemo = 'failed'; button.title = `${DEMO_TEXT.failed} (${err?.message ?? err})`; }
      cache.delete(key);
      return null;
    });
  }
  function closeNow() {
    win.clearTimeout(openTimer); win.clearTimeout(closeTimer); wanted++;
    if (current) current.hide();
    current = null; pinned = false; overPop = false;
    button.setAttribute('aria-expanded', 'false'); button.removeAttribute('aria-controls');
    if (button.dataset.sdemo !== 'failed') button.dataset.sdemo = '';
  }
  // Closes a moment later unless it is pinned (clicked), the pointer is over the "?" or the popover, or the "?" has focus.
  const keepOpen = () => pinned || overButton || overPop || doc.activeElement === button;
  function closeSoon() { win.clearTimeout(openTimer); win.clearTimeout(closeTimer); closeTimer = win.setTimeout(() => { if (!keepOpen()) closeNow(); }, CLOSE_DELAY_MS); }
  on(button, 'pointerenter', e => {
    if (e.pointerType === 'touch') return;
    overButton = true; win.clearTimeout(openTimer);
    openTimer = win.setTimeout(() => open(), OPEN_DELAY_MS);
  });
  on(button, 'pointerleave', e => { overButton = false; if (e.pointerType !== 'touch') closeSoon(); });
  on(button, 'focus', () => open());
  on(button, 'blur', () => closeSoon());
  on(button, 'click', e => {
    e.stopPropagation?.();
    if (current && current.isOpen && pinned) { closeNow(); return; }
    pinned = true; open();
  });
  on(doc, 'keydown', e => { if (e.key === 'Escape' && current) { closeNow(); button.focus?.({ preventScroll: true }); } });
  on(doc, 'pointerdown', e => { if (current && pinned && !current.el?.contains(e.target) && e.target !== button) closeNow(); });
  on(win, 'resize', () => { if (current?.isOpen) { current.size(); place(current); current.run(); } });
  on(win, 'scroll', () => { if (current?.isOpen) place(current); }, true);
  const show = () => { button.hidden = !keyOf(flag); };
  show();
  return {
    setFlag(next) { if (next === flag) return; flag = next ?? null; if (current) closeNow(); show(); },
    open, close: closeNow,
    isOpen: () => !!current?.isOpen,
    get current() { return current; },
    dispose() {
      closeNow();
      for (const [node, type, fn, opts] of listeners) node.removeEventListener(type, fn, opts);
      listeners.length = 0;
      for (const p of cache.values()) p.then(d => d?.remove(), () => {});
      cache.clear();
    },
  };
}
