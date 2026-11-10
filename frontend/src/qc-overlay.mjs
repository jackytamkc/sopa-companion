// The mixing check on the stage's 2-D overlay canvas: the soft per-bin layer that
// stands in for the cells under Colour by: Mixing while single cells are too small to carry the colours (the whole
// sample in view; in Outlines mode, until the outlines are drawn), and the legend beside the stage. The layer is each
// bin's 3 x 3 window share (sum M / sum J: a few cells cannot light up a bin), drawn as strongly as the bin is full (a
// bin with a few cells at the tissue's edge fades out instead of lighting up background), and faint where the share is
// at or below the tissue's usual level, so only patches above it carry colour and the tissue stays visible.
//
// No flagged area is drawn (they added little on screen): no outline, wash or
// letter chip, nothing to hover or click, and no outline key in the legend. The engine still flags areas for its
// suggestions and trials; a card's "Look at its cells" moves the view there (main.js), with nothing drawn.
//
// The overlay is drawn by main.js's drawOverlay(), which calls draw() after its own region outline; drawOverlay already
// runs on every camera change. The per-bin layer's cross-fade asks for its few frames through the stage's scheduler
// (Stage.addAnimator): the overlay changes in the stage's own animation frames, at most 3 per result, and nothing is
// scheduled while idle. A layer drawn for options that have changed since (Colour by, the display mode: no camera move)
// gets one refresh frame the same way. In 3-D the overlay is hidden and the cell colours carry the picture.
//
// Two marks are drawn here too, dashed over a dark casing and never shaded by Go deeper's focus: a test of a setting's
// square while the test exists (setTestSquare), and Go
// deeper's starting outline of its cell while its Show is on (Stage.setStartOutline: the stage keeps it, this draws it).
// A change of either gets one refresh frame the same way; both are drawn whether or not the mixing check's own layer is
// (they belong to the test and the window), in 2-D.
//
// Every word on screen comes from qc-copy.mjs (the plain-language deck, checked for banned words).
import { QC_TEXT as C, copy, number } from './qc-copy.mjs';
import { MIXING_COLORS as M, MIXING_CODES, tintPlan, tintValue, tintNext } from './stage.js';
import { harness } from './harness-hook.mjs';
import { deep } from './qc-deeper-copy.mjs';

export const FONT = '600 12px "Ubuntu Sans", Inter, system-ui, sans-serif';   // the words drawn here (Go deeper's arrows)
export const OVERVIEW_SCALE = Object.freeze({ full: .25, none: .6 });   // CSS px per µm: the per-bin layer is drawn fully
                                                                        // at or below `full` (a 10 µm cell ≤ 2.5 px)
                                                                        // and not at all from `none` (6 px) on
// In Outlines mode the cells are their outlines, which stage.js draws at most 0.34 of a voxel wide: under
// about 2.4 device px per voxel they are not drawn at all, so single cells cannot carry the colours until then. Device
// px per voxel: the per-bin layer is drawn fully at or below `full` and not at all from `none` on.
export const OUTLINE_VOXEL_PX = Object.freeze({ full: 1.2, none: 2.6 });
export const LAYER_ALPHA = .85;       // the per-bin layer at full strength
export const LOW_ALPHA = .2;          // ... a patch at or below the tissue's usual level (rising to full at the next step)
export const FULL_J = 20;             // judged cells that make a 100 µm bin full (scaled by its area), at most the grid's
                                      // typical bin (the median of the bins holding cells: denser samples, finer bins)
// The marks (CSS px): a test's square, light, in 7/5 px dashes; Go deeper's starting outline, white in finer dashes (an
// outline of what was, beside the cell's own solid one).
export const DASHES = Object.freeze([7, 5]);
export const SQUARE_STYLE = Object.freeze({ color: 'rgba(230,234,240,.85)', width: 1.5, dashes: DASHES });
export const START_STYLE = Object.freeze({ color: '#ffffff', width: 1.5, dashes: Object.freeze([4, 3]) });
const CASING = 'rgba(11,14,19,.55)';  // the dark line under every line of the overlay, so it reads over any colour
// Share of a group's judged cells that look mixed -> colour of the per-bin layer, on the cells' own one-hue ramp,
// anchored on the tissue's own level P (binStops): little or none at or below P, some at the margin an area needs to be
// flagged (share >= max(1.5 P, P + 4 points)), a lot at twice that margin. Groups as mixed as a flagged area
// light up wherever the tissue's level is (a whole sample can read 13-15 in 100, so absolute steps would colour all of
// it "some"), and a shift of the model's overall level cannot light the whole tissue. Without a level: absolute steps.
export const BIN_STOPS = Object.freeze([[0, M.little], [.15, M.some], [.3, M.mixed]]);
export const BIN_MARGIN = Object.freeze({ mult: .5, add: .04 });   // the areas' enter margin over P
/** The ramp's stops for tissue level `level` (share 0-1, e.g. the headline / 100), or BIN_STOPS without one. */
export function binStops(level) {
  if (!Number.isFinite(level) || level < 0) return BIN_STOPS;
  const margin = Math.max(BIN_MARGIN.mult * level, BIN_MARGIN.add);
  return [[level, M.little], [level + margin, M.some], [level + 2 * margin, M.mixed]];
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
export const hexRgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
export const rgba = (hex, a) => `rgba(${hexRgb(hex).join(',')},${a})`;

/** How strongly the per-bin layer covers the cells at `scale` CSS px per µm: 1 at the whole-sample overview, fading to 0
 *  as cells grow large enough to read one by one. `mode` (stage options.mode: 0 cells, 1 outlines), `voxel` (µm, of
 *  the raster shown) and `ratio` (device px per CSS px): with outlines only, until they are drawn (OUTLINE_VOXEL_PX). */
export function binsWeight(scale, { mode = 0, voxel = 1, ratio = 1 } = {}) {
  if (!Number.isFinite(scale)) return 0;
  const cells = 1 - smoothstep(OVERVIEW_SCALE.full, OVERVIEW_SCALE.none, scale);
  if (mode !== 1) return cells;
  const px = scale * (ratio > 0 ? ratio : 1) * (voxel > 0 ? voxel : 1);
  return Math.max(cells, 1 - smoothstep(OUTLINE_VOXEL_PX.full, OUTLINE_VOXEL_PX.none, px));
}
/** The overview: the per-bin layer stands in for single cells (binsWeight >= 1/2, same options), so the legend shows the
 *  patches' own key. */
export function isOverview(scale, options) { return binsWeight(scale, options) >= .5; }

/** A bins grid from GET /runs/{id}/qc/{seq}/bins: `header` is its X-Bins value {x0, y0, bin (µm), nx, ny, planes?}
 *  (JSON text or the object), `buffer` `planes` planes of one byte per bin, row-major (by * nx + bx): the bin's 3 x 3
 *  window share, coded as the cells are (floor(share x 250), 255 = nothing judged there), then (planes 2) the bin's own
 *  judged cells (at most 254). A result stored before the second plane has the first only (planes 1: each bin's own
 *  share, smoothShares averages it). */
export function decodeBins(buffer, header) {
  const meta = typeof header === 'string' ? JSON.parse(header) : header;
  if (!meta) throw new Error('Missing X-Bins geometry');
  const { x0, y0, bin, nx, ny } = meta, planes = meta.planes ?? 1, values = new Uint8Array(buffer);
  if (![x0, y0, bin].every(Number.isFinite) || !(bin > 0) || !Number.isInteger(nx) || !Number.isInteger(ny) || nx <= 0 || ny <= 0
    || !(planes === 1 || planes === 2) || values.length !== nx * ny * planes) throw new Error('Truncated or invalid bins grid');
  return { x0, y0, bin, nx, ny, planes, values };
}
/** The bins of frame `seq` of `run` (decodeBins), or null while the engine has none for it (404). `seq` 'start': the
 *  start line's own (GET /runs/{id}/qc/start/bins), for the frame of the starting cells. */
export async function fetchBins(run, seq, { signal, fetch = globalThis.fetch } = {}) {
  const r = await fetch(`/api/v1/runs/${encodeURIComponent(run)}/qc/${encodeURIComponent(seq)}/bins`, { signal, credentials: 'same-origin' });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Mixing bins: HTTP ${r.status}`);
  return decodeBins(await r.arrayBuffer(), r.headers.get('X-Bins'));
}
/** RGBA of a share (0-1) of judged cells that look mixed on the ramp `stops` (binStops); transparent for NaN (nothing
 *  judged). */
export function shareColor(share, stops = BIN_STOPS) {
  if (!Number.isFinite(share)) return [0, 0, 0, 0];
  let k = 1;
  while (k < stops.length - 1 && share > stops[k][0]) k++;
  const [s0, c0] = stops[k - 1], [s1, c1] = stops[k], t = clamp((share - s0) / (s1 - s0), 0, 1);
  const a = hexRgb(c0), b = hexRgb(c1);
  return [...a.map((v, i) => Math.round(v + (b[i] - v) * t)), 255];
}
/** Each bin's share (0-1; NaN where nothing is judged) averaged with its judged neighbours: the 3 x 3 bins (300 µm on
 *  whole samples, 150 µm in detail runs) the engine flags its areas on. One 100 µm bin holds only some 10-25 judged
 *  cells, so its own share wobbles by about 8 points, and a bin at the tissue's edge with one or two cells reads 0 or
 *  50 %. */
export function smoothShares(grid) {
  const { nx, ny, values } = grid, out = new Float32Array(nx * ny), none = MIXING_CODES.unjudged;
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = y * nx + x;
    if (!(values[i] < none)) { out[i] = NaN; continue; }
    let sum = 0, n = 0;
    for (let v = Math.max(0, y - 1); v <= Math.min(ny - 1, y + 1); v++) for (let u = Math.max(0, x - 1); u <= Math.min(nx - 1, x + 1); u++) {
      const c = values[v * nx + u];
      if (c < none) { sum += c; n++; }
    }
    out[i] = sum / n / MIXING_CODES.scale;
  }
  return out;
}
/** The judged cells of a full bin of `grid`: FULL_J scaled to the bin's area, at most the median of the bins holding
 *  cells (a sparse sample's typical bin is full too). */
export function fullJudged(grid) {
  const n = grid.nx * grid.ny, held = [];
  for (let i = 0; i < n; i++) if (grid.values[n + i] >= 1) held.push(grid.values[n + i]);
  held.sort((a, b) => a - b);
  const median = held.length ? held[held.length >> 1] : 1;
  return Math.max(1, Math.min(FULL_J * (grid.bin / 100) ** 2, median));
}
/** Each bin's share (0-1; NaN where nothing is judged) and how full it is (0-1: its judged cells over a full bin's,
 *  squared, so a bin half outside the tissue shows at a quarter: the layer does not glow over background; 1 for a grid
 *  without the second plane): the window share the engine computed (planes 2), else the bins' own shares averaged here
 *  (smoothShares). */
export function layerValues(grid) {
  const n = grid.nx * grid.ny, fullness = new Float32Array(n);
  if ((grid.planes ?? 1) < 2) {
    const share = smoothShares(grid);
    for (let i = 0; i < n; i++) fullness[i] = Number.isFinite(share[i]) ? 1 : 0;
    return { share, fullness };
  }
  const share = new Float32Array(n), full = fullJudged(grid), none = MIXING_CODES.unjudged;
  for (let i = 0; i < n; i++) {
    const code = grid.values[i], judged = grid.values[n + i];
    share[i] = code < none && judged >= 1 ? code / MIXING_CODES.scale : NaN;
    fullness[i] = Number.isFinite(share[i]) ? Math.min(1, judged / full) ** 2 : 0;
  }
  return { share, fullness };
}
/** How strongly a share shows on the ramp `stops`: LOW_ALPHA at or below its first step (the tissue's usual level),
 *  rising to 1 at the next. */
export function shareStrength(share, stops = BIN_STOPS) {
  if (!Number.isFinite(share)) return 0;
  const [s0] = stops[0], [s1] = stops[1];
  return LOW_ALPHA + (1 - LOW_ALPHA) * clamp((share - s0) / Math.max(1e-6, s1 - s0), 0, 1);
}
const VALUES = new WeakMap();
const valuesOf = grid => { let v = VALUES.get(grid); if (!v) VALUES.set(grid, v = layerValues(grid)); return v; };
const sameGrid = (a, b) => !!a && !!b && ['x0', 'y0', 'bin', 'nx', 'ny'].every(k => a[k] === b[k]);
/** One bin's RGBA: its share's colour on the ramp `stops`, as strong as it is full and as far above the usual level. */
function binRgba(values, i, stops) {
  const share = values.share[i], c = shareColor(share, stops);
  c[3] = Math.round(255 * values.fullness[i] * shareStrength(share, stops));
  return c;
}
/** The per-bin layer's pixels (nx x ny RGBA) for grid `to` (layerValues) on the ramp `stops`, cross-faded from `from`
 *  (same geometry) by `t` (premultiplied, so a bin appearing or going fades without darkening). */
export function binsPixels(to, from = null, t = 1, stops = BIN_STOPS) {
  const n = to.nx * to.ny, out = new Uint8ClampedArray(n * 4), blend = from && t < 1 && sameGrid(from, to);
  const now = valuesOf(to), before = blend ? valuesOf(from) : null;
  for (let i = 0; i < n; i++) {
    const b = binRgba(now, i, stops);
    if (!blend) { out.set(b, i * 4); continue; }
    const a = binRgba(before, i, stops), alpha = a[3] * (1 - t) + b[3] * t;
    for (let k = 0; k < 3; k++) out[i * 4 + k] = alpha > 0 ? (a[k] * a[3] * (1 - t) + b[k] * b[3] * t) / alpha : 0;
    out[i * 4 + 3] = alpha;
  }
  return out;
}
/** The bins a per-bin image colours (alpha > 0) as one path of whole-bin rectangles in bin units (rows of runs), to clip
 *  the magnified, smoothed image to; null where Path2D is not there (tests). */
export function occupiedPath(data, nx, ny) {
  if (typeof globalThis.Path2D !== 'function') return null;
  const path = new globalThis.Path2D();
  let any = false;
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx;) {
      if (!(data[(y * nx + x) * 4 + 3] > 0)) { x++; continue; }
      const from = x;
      while (x < nx && data[(y * nx + x) * 4 + 3] > 0) x++;
      path.rect(from, y, x - from, 1); any = true;
    }
  }
  return any ? path : null;
}
function makeCanvas(width, height) {
  if (typeof OffscreenCanvas === 'function') return new OffscreenCanvas(width, height);
  const canvas = globalThis.document.createElement('canvas'); canvas.width = width; canvas.height = height; return canvas;
}
/** A pill outline (left, top, width, height): the arrows' labels. */
function pill(g, x, y, w, h) {
  const r = h / 2;
  g.beginPath(); g.moveTo(x + r, y); g.lineTo(x + w - r, y); g.arc(x + w - r, y + r, r, -Math.PI / 2, Math.PI / 2);
  g.lineTo(x + r, y + h); g.arc(x + r, y + r, r, Math.PI / 2, Math.PI * 3 / 2); g.closePath();
}
/** When a cross-fade (stage.js tintPlan, started at plan.at) next needs a frame, for an animator that last drew at
 *  `drawnAt` (absolute ms); null once it is complete. */
function fadeDue(plan, drawnAt) {
  if (!(drawnAt >= plan.at)) return plan.at;
  const next = tintNext(plan, drawnAt - plan.at);
  return next == null ? null : plan.at + next;
}

/**
 * The mixing check on the overlay: the per-bin layer and the two marks. `stage`: the Stage (toScreen, scale, ratio,
 * orbit, options.cellColorBy and .mode, frame.voxel, startOutline, addAnimator/replan); options: redraw() (main.js's
 * drawOverlay: asked for the motion and refresh frames), now(), reducedMotion() -> bool, createCanvas(w, h).
 */
export class QcOverlay {
  constructor(stage, { redraw = () => {}, now = () => performance.now(), reducedMotion = null, createCanvas = makeCanvas } = {}) {
    this.stage = stage; this.redraw = redraw; this.now = now; this.createCanvas = createCanvas;
    this.prefersReduced = reducedMotion; this.calm = false;   // calm: the panel's "calm motion" switch
    this.visible = true; this.anchorPer100 = null;
    this.bins = null; this.binsFrom = null; this.binsPlan = null; this.binsAt = -Infinity; this.images = new Map();
    this.drawnAt = -Infinity;
    this.drawnWeight = 0; this.drawnWants = false;   // the per-bin layer as last drawn (refreshDue)
    this.square = null;                              // a test's square ([x0, y0, x1, y1] µm: setTestSquare) or null
    this.drawnSquare = null; this.drawnStart = null; // the marks as last drawn (marksDue)
    stage.addAnimator?.(this);
  }
  /** No motion: prefers-reduced-motion or the calm switch. */
  reduced() {
    if (this.calm) return true;
    if (this.prefersReduced) return !!this.prefersReduced();
    try { return !!(this.motionQuery ??= globalThis.matchMedia?.('(prefers-reduced-motion: reduce)'))?.matches; } catch { return false; }
  }
  /** Mixing colours on the cells (Layers > Cells > Colour by: Mixing). */
  mixingShown() { return this.stage.options?.cellColorBy === 'mixing'; }
  /** Drawn at all: shown (`visible`, set by main.js) and in 2-D. */
  drawable() { return this.visible && !this.stage.orbit?.active; }
  /** How strongly the per-bin layer covers the cells now (binsWeight for the stage's zoom, display mode and raster). */
  binsWeight() {
    const s = this.stage;
    return binsWeight(s.scale, { mode: s.options?.mode, voxel: s.frame?.voxel, ratio: s.ratio });
  }
  /** Whether the per-bin layer stands in for single cells now (with Colour by: Mixing, in 2-D): the legend shows the
   *  patches' own key (MixingLegend.update's `overview`). */
  overview() { return this.drawable() && this.mixingShown() && this.binsWeight() >= .5; }

  // ---- content -----------------------------------------------------------------------------------------------------
  /** The tissue's usual level the per-bin layer's ramp is anchored on (binStops): `anchorPer100` the start line's (fixed
   *  for the run, so the colours change only where the tissue changed), else `tissuePer100` (the level shown, e.g.
   *  summary.shown: a run from nuclei only has no start line). Returns whether the overlay must be drawn again: a new
   *  level recolours a layer on screen. */
  setLevel({ anchorPer100 = null, tissuePer100 = null } = {}) {
    const anchor = Number.isFinite(anchorPer100) ? anchorPer100 : Number.isFinite(tissuePer100) ? tissuePer100 : null;
    if (anchor === this.anchorPer100) return false;
    const recolour = !!this.bins && this.drawnWeight > 0;
    this.anchorPer100 = anchor;
    return recolour;
  }
  /** The per-bin layer's ramp now: anchored on the tissue's usual level (binStops; the start line's when known). */
  binStops() { return binStops(this.anchorPer100 == null ? null : this.anchorPer100 / 100); }
  /** The per-bin layer's grid for the frame shown (decodeBins), or null. Cross-faded like the cell tint (stage.js
   *  tintPlan: only a second or more after the previous one; never with motion false). Returns whether it changed. */
  setBins(grid, { at = this.now(), motion = true } = {}) {
    grid = grid || null;
    if (grid === this.bins) return false;
    const plan = grid && sameGrid(grid, this.bins) && this.mixingShown() ? tintPlan(at - this.binsAt, !motion || this.reduced()) : null;
    this.binsFrom = plan ? this.bins : null; this.bins = grid; this.binsAt = grid ? at : -Infinity;
    this.binsPlan = plan ? { ...plan, at } : null; this.images.clear();
    if (plan) this.stage.replan?.();
    return true;
  }
  /** Whether main.js should fetch the bins of the frame shown: Mixing colours are on, in 2-D, zoomed out far enough
   *  for the per-bin layer to show. */
  wantsBins() { return this.drawable() && this.mixingShown() && this.binsWeight() > 0; }
  /** Forget the layer (another run, the check switched off). */
  clear() {
    const had = !!this.bins;
    this.bins = this.binsFrom = this.binsPlan = null; this.binsAt = -Infinity; this.images.clear();
    return had;
  }
  /** A test of a setting's square while the test exists: {box: [x0, y0, x1, y1] µm}
   *  or null (also for a box that is not one). Drawn dashed in the stage's next frame (a refresh, not motion), in 2-D.
   *  Returns whether it changed. */
  setTestSquare(sq) {
    const b = sq?.box, ok = Array.isArray(b) && b.length === 4 && b.every(Number.isFinite) && b[2] > b[0] && b[3] > b[1];
    if (ok ? !!this.square && b.every((v, i) => v === this.square[i]) : !this.square) return false;
    this.square = ok ? Object.freeze([...b]) : null;
    this.stage.replan?.();
    return true;
  }
  /** The test square as set ([x0, y0, x1, y1] µm), or null: the harness hook's. */
  get testSquare() { return this.square; }

  // ---- motion: the stage's animator ---------------------------------------------------------------------------------
  /** The per-bin layer's strength as it should be drawn now: 0 without Mixing colours. */
  layerWeight() { return this.mixingShown() ? this.binsWeight() : 0; }
  /** When the next frame of the per-bin cross-fade is due (absolute ms), or null. */
  motionDue() {
    const plan = this.binsPlan;
    return plan && this.bins && this.mixingShown() && this.binsWeight() > 0 ? fadeDue(plan, this.drawnAt) : null;
  }
  /** Whether the per-bin layer on screen is out of date without any camera move (which redraws the overlay anyway):
   *  Colour by or the display mode changed its strength, or it is wanted now and not fetched yet (drawing again lets
   *  main.js's drawOverlay fetch it). One refresh, no motion. */
  refreshDue() {
    return (!!this.bins && Math.abs(this.layerWeight() - this.drawnWeight) > 1e-3) || (!this.drawnWants && this.wantsBins());
  }
  /** Whether the marks on screen are not the ones set: the test square, or the stage's starting outline (one refresh). */
  marksDue() { return this.square !== this.drawnSquare || (this.stage.startOutline ?? null) !== this.drawnStart; }
  /** The stage's animator: when the next frame is due (absolute ms), or null. Nothing is due while idle. */
  due(now = this.now()) {
    if (this.stage.orbit?.active) return null;
    if (this.marksDue()) return now;   // drawn with the check's own layer or without it (drawMarks)
    if (!this.drawable()) return null;
    const motion = this.motionDue();
    return this.refreshDue() ? Math.min(now, motion ?? now) : motion;
  }
  /** One frame, in the stage's animation frame. Returns whether it was motion (a refresh is not: stage.js counts only
   *  motion frames for the harness). */
  frame(now = this.now()) {
    const due = this.motionDue(), moving = due != null && due <= now;
    this.drawnAt = now; this.drawnWeight = this.layerWeight(); this.drawnWants = this.wantsBins();
    this.drawnSquare = this.square; this.drawnStart = this.stage.startOutline ?? null;
    this.redraw();
    return moving;
  }

  // ---- drawing -----------------------------------------------------------------------------------------------------
  /** Draw on the overlay's 2-D context `g` (transformed to CSS px): the per-bin layer, then the marks. */
  draw(g) {
    const now = this.now();
    this.drawnAt = now; this.drawnWeight = this.layerWeight(); this.drawnWants = this.wantsBins();
    this.drawnSquare = this.square; this.drawnStart = this.stage.startOutline ?? null;
    if (!this.drawable()) { if (!this.stage.orbit?.active) this.drawMarks(g); return; }
    g.save();
    if (this.drawnWeight > 0 && this.bins) this.drawBins(g, this.drawnWeight, now);
    this.drawMarks(g);
    g.restore();
  }
  /** The marks (SQUARE_STYLE, START_STYLE): the test square and the stage's starting outline of Go deeper's cell, each a
   *  dashed line over a solid dark casing, at full strength (the focus never shades them). */
  drawMarks(g) {
    const s = this.stage, start = s.startOutline, square = this.square;
    if (!square && !(start?.count >= 2)) return;
    const line = (style, trace) => {
      g.setLineDash([]); trace(); g.strokeStyle = CASING; g.lineWidth = style.width + 1.5; g.stroke();
      g.setLineDash(style.dashes); g.strokeStyle = style.color; g.lineWidth = style.width; g.stroke();
    };
    g.save(); g.globalAlpha = 1; g.lineJoin = 'round'; g.lineCap = 'butt';
    if (square) {
      const [x0, y0] = s.toScreen(square[0], square[1]), [x1, y1] = s.toScreen(square[2], square[3]);
      line(SQUARE_STYLE, () => { g.beginPath(); g.rect(x0, y0, x1 - x0, y1 - y0); });
    }
    if (start?.count >= 2) {
      const p = start.points;
      line(START_STYLE, () => {
        g.beginPath();
        for (let i = 0; i < start.count; i++) { const [x, y] = s.toScreen(p[2 * i], p[2 * i + 1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
        if (start.closed) g.closePath();
      });
    }
    g.setLineDash([]); g.restore();
  }
  drawBins(g, weight, now) {
    const grid = this.bins, plan = this.binsPlan, t = plan ? tintValue(plan, now - plan.at) : 1, step = t < 1 && this.binsFrom ? t : 1;
    const key = `${step}|${this.anchorPer100}`;
    let image = this.images.get(key);
    if (!image) {
      image = this.createCanvas(grid.nx, grid.ny);
      const c = image.getContext('2d'), pixels = c.createImageData(grid.nx, grid.ny);
      const data = binsPixels(grid, step < 1 ? this.binsFrom : null, step, this.binStops());
      pixels.data.set(data); c.putImageData(pixels, 0, 0);
      image.occupied = occupiedPath(data, grid.nx, grid.ny);
      if (this.images.size >= 8) this.images.clear();   // a few steps of one grid; a new level starts again
      this.images.set(key, image);
    }
    const [x0, y0] = this.stage.toScreen(grid.x0, grid.y0), [x1, y1] = this.stage.toScreen(grid.x0 + grid.nx * grid.bin, grid.y0 + grid.ny * grid.bin);
    g.globalAlpha = weight * LAYER_ALPHA; g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    if (image.occupied) {
      // Smooth inside the tissue, but never past its bins: drawn magnified with smoothing, a patch at the tissue's edge
      // spilled its colour tens of µm onto the empty background.
      g.save(); g.translate(x0, y0); g.scale((x1 - x0) / grid.nx, (y1 - y0) / grid.ny); g.clip(image.occupied);
      g.drawImage(image, 0, 0, grid.nx, grid.ny); g.restore();
    } else g.drawImage(image, x0, y0, x1 - x0, y1 - y0);
    g.globalAlpha = 1;
  }
  dispose() { this.stage.removeAnimator?.(this); this.clear(); }
}

// ---- the legend (bottom left of the stage) ----------------------------------------------------------------------------
const LEGEND_CSS = 'position:absolute;left:12px;bottom:12px;max-width:290px;padding:7px 10px;border-radius:var(--r-2,6px);'
  + 'background:rgba(14,17,22,.88);border:1px solid var(--hair,#2a313c);color:var(--ink-1,#e6eaf0);font-size:var(--fs-1,11px);'
  + 'line-height:1.45;pointer-events:none;z-index:2';
// Its place (CSS px from the stage's bottom left, as LEGEND_CSS has it), the most it covers (its widest with padding and
// border; its tallest key, the cells' title and four rows: 99 px at 11 px type, with a little room), and its gap above
// the test corner when it moves out of the corner's way (dock).
export const LEGEND_PLACE = Object.freeze({ left: 12, bottom: 12, width: 312, height: 104, gap: 8 });
const SWATCH_CSS = 'display:inline-block;flex:none;width:14px;height:10px;border-radius:2px';
/** Diagonal stripes of the "not judged" grey, as the stage draws those cells. */
export const UNJUDGED_SWATCH = `repeating-linear-gradient(45deg,${rgba(M.unjudged, .75)} 0 3px,${rgba(M.unjudged, .2)} 3px 6px)`;

/**
 * The legend of Colour by: Mixing, one row per colour shade, each saying in a few words what it means:
 * new MixingLegend(host) appends it (hidden) to `host` (the stage's wrapper);
 * update({mixing, overview, view3d, stops, noValues}) shows what applies: the cells' key (a short title, then under
 * 10 %, 10-30 %, 30 % or more and too few transcripts: the stripes), or zoomed out to groups of cells the patches' own key
 * the same way (a short title and its three rows: the numbers of the ramp in use, `stops`), "Not checked yet" while no
 * values colour the cells. No note line and no outline key: no area is drawn.
 */
export class MixingLegend {
  constructor(host, { doc = host?.ownerDocument ?? globalThis.document } = {}) {
    const el = (tag, css = '', text = '') => { const n = doc.createElement(tag); if (css) n.style.cssText = css; if (text) n.textContent = text; return n; };
    const row = (swatch, text) => { const r = el('div', 'display:flex;align-items:center;gap:6px'); r.append(swatch, el('span', '', text)); return r; };
    const swatch = background => { const s = el('span', SWATCH_CSS); s.style.background = background; return s; };
    this.root = el('div', LEGEND_CSS); this.root.hidden = true;
    this.root.setAttribute?.('role', 'note'); this.root.className = 'qc-legend';
    // A cell's own share (zoomed in): the cells' four shades.
    this.colours = el('div');
    this.unjudged = swatch(UNJUDGED_SWATCH); this.flatUnjudged = false;
    this.steps = el('div');
    this.steps.append(row(swatch(M.little), C.legendLittle), row(swatch(M.some), C.legendSome), row(swatch(M.mixed), C.legendMixed),
      row(this.unjudged, C.legendUnjudged));
    this.noValues = row(swatch(rgba(M.unjudged, .5)), C.legendNoValues); this.noValues.hidden = true;
    this.colours.append(el('div', 'font-weight:600;margin-bottom:3px', C.legendTitle), this.steps, this.noValues);
    // Zoomed out: each patch's share of cells that look mixed, against the tissue's usual level (qc-overlay's layer).
    this.patches = el('div'); this.patches.hidden = true;
    this.patchRows = [M.little, M.some, M.mixed].map((colour, i) => {
      const text = el('span'), r = el('div', 'display:flex;align-items:center;gap:6px');
      r.append(swatch(i ? colour : rgba(colour, LOW_ALPHA + .15)), text);   // the usual level: as faint as on the stage
      return { row: r, text };
    });
    this.patches.append(el('div', 'font-weight:600;margin-bottom:3px', C.overviewTitle), ...this.patchRows.map(r => r.row));
    this.root.append(this.colours, this.patches);
    host?.append(this.root);
  }
  /** mixing: Colour by: Mixing is on; overview: zoomed out to groups of cells (isOverview); view3d: the 3-D view (its
   *  "not judged" cells are plain grey, no stripes; no patches); stops: the patches' ramp (QcOverlay.binStops(): its
   *  first three steps as shares 0-1) for the overview's key; noValues: no values colour the cells yet (they are plain
   *  grey). Hidden without Mixing colours: there is nothing else to explain. */
  update({ mixing = false, overview = false, view3d = false, stops = null, noValues = false } = {}) {
    const set = (node, key, value) => { if (node[key] !== value) node[key] = value; };
    const patches = mixing && overview && !view3d && !noValues && Array.isArray(stops) && stops.length >= 3;
    set(this.colours, 'hidden', !mixing || patches);
    set(this.steps, 'hidden', noValues);
    set(this.noValues, 'hidden', !noValues);
    // Kept here, not compared with the style (the browser rewrites a gradient's text): no style write per redraw.
    if (this.flatUnjudged !== view3d) { this.flatUnjudged = view3d; this.unjudged.style.background = view3d ? rgba(M.unjudged, .5) : UNJUDGED_SWATCH; }
    set(this.patches, 'hidden', !patches);
    if (patches) {
      const [a, b, c] = stops.map(([share]) => number(Math.round(share * 100)));
      set(this.patchRows[0].text, 'textContent', copy('overviewLow', { a }));
      set(this.patchRows[1].text, 'textContent', copy('overviewMid', { b }));
      set(this.patchRows[2].text, 'textContent', copy('overviewHigh', { c }));
    }
    set(this.root, 'hidden', this.colours.hidden && this.patches.hidden);
  }
  /** Every word the legend can show (for the banned-word check). */
  texts() {
    return [C.legendTitle, C.legendLittle, C.legendSome, C.legendMixed, C.legendUnjudged, C.legendNoValues, C.overviewTitle,
      ...this.patchRows.map(r => r.text.textContent).filter(Boolean)];
  }
  /** Out of the test corner's way ("the legend moves above the corner while it is open"): `corner`
   *  the corner's box as it reports it ({left, bottom, w, h} CSS px in the stage) or null; `height` the legend's own
   *  (CSS px, as the browser reports it; at most LEGEND_PLACE.height without one). Above the corner while the corner
   *  lies where the legend sits, else in its own place; at once (no motion). Returns its bottom (CSS px). */
  dock(corner = null, height = LEGEND_PLACE.height) {
    const P = LEGEND_PLACE, c = corner && corner.w > 0 && corner.h > 0 && Number.isFinite(corner.left) && Number.isFinite(corner.bottom) ? corner : null;
    const tall = height > 0 ? Math.min(height, P.height) : P.height;
    const under = !!c && c.left < P.left + P.width && c.left + c.w > P.left && c.bottom < P.bottom + tall;
    const bottom = under ? Math.round(c.bottom + c.h + P.gap) : P.bottom;
    if (this.root.style.bottom !== `${bottom}px`) this.root.style.bottom = `${bottom}px`;
    return bottom;
  }
  dispose() { this.root.remove?.(); }
}

// ---- Go deeper's arrows (qc-deeper.js) ------------------------------------------------------------------------------
// From each neighbour the window's cell's RNA seems to come from (a source of at least ARROW_MIN_SHARE of its RNA, not
// one that looks like its own kind: those get the stage's grey dotted outline and no arrow) to the cell, on the overlay in
// world µm (they follow pan and zoom): slightly curved, an arrowhead at the cell, magenta on a dark halo, 1.5-5 px wide by
// share, the top ARROW_LABELS labelled on dark pills. Only while cells are at least ARROW_MIN_PX across (else the window
// says to zoom in) and in 2-D. A new source draws on from the neighbour over ARROW_DRAW_MS with one pulse ring, only the
// first time its share passes ARROW_PULSE_SHARE (and every arrow when the window moves to another cell); a leaving one
// fades. Its frames come through the stage's scheduler on the 150 ms steps (Stage.addAnimator), at most 4 per update, and
// nothing is scheduled while nothing changes.
export const ARROWS = 5, ARROW_LABELS = 3, ARROW_MIN_SHARE = .01, ARROW_PULSE_SHARE = .05, ARROW_MIN_PX = 6;
export const ARROW_STEP_MS = 150, ARROW_DRAW_MS = 450, ARROW_FADE_MS = 300, ARROW_HIT_PX = 6;
export const ARROW_HALO = '#0e1116';
/** An arrow's width (CSS px) for a share (0-1) of the cell's RNA. */
export const arrowWidth = share => 1.5 + 3.5 * clamp((Number.isFinite(share) ? share : 0) / .4, 0, 1);
/** The progress (0-1) of a draw-on or fade started `elapsed` ms ago, on the stage's steps: the first paint is already one
 *  step in (1/4, 1/2, 3/4, 1 over 450 ms). */
export function arrowProgress(elapsed, duration) { return clamp((Math.max(0, elapsed) + ARROW_STEP_MS) / (duration + ARROW_STEP_MS), 0, 1); }
/** When a motion started at `start` next needs a frame after `drawnAt` (its steps), or null once it is complete. */
function stepDue(start, duration, drawnAt) {
  if (!Number.isFinite(start) || drawnAt >= start + duration) return null;
  const k = Math.max(1, Math.floor((drawnAt - start) / ARROW_STEP_MS) + 1);
  const t = start + k * ARROW_STEP_MS;
  return t <= start + duration ? t : null;
}
/** The arrow's curve in screen px from `a` (the neighbour) to `b` (the cell): a quadratic bend of `bend` of its length;
 *  at(t) its point, dir(t) its direction. */
export function arrowCurve(a, b, bend = .15) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
  const c = [(a[0] + b[0]) / 2 - dy * bend, (a[1] + b[1]) / 2 + dx * bend];
  const at = t => { const u = 1 - t; return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]]; };
  const dir = t => { const u = 1 - t, x = 2 * u * (c[0] - a[0]) + 2 * t * (b[0] - c[0]), y = 2 * u * (c[1] - a[1]) + 2 * t * (b[1] - c[1]), n = Math.hypot(x, y) || 1; return [x / n, y / n]; };
  return { a, b, c, len, at, dir };
}
/** The part of `curve` from its start to t (de Casteljau). */
function tracePart(g, curve, t) {
  const { a, c } = curve, q = [a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t], end = curve.at(t);
  g.beginPath(); g.moveTo(a[0], a[1]); g.quadraticCurveTo(q[0], q[1], end[0], end[1]);
}
/**
 * The arrows of Go deeper's window. new DeeperArrows(stage, {redraw, now, reducedMotion, onNote}); set(focus) with
 * {cell, target: [x, y] µm, sources: [{label, share (0-1), from: [x, y] µm, name}], hot (a row hovered or pinned: 1 px
 * wider), diameter (the cell's, µm)} or null; draw(g) from main.js's drawOverlay; hit(x, y) -> the source's label under a
 * CSS point. onNote(kind): what the window should say about them ('' drawn, 'zoom', '3d').
 */
export class DeeperArrows {
  constructor(stage, { redraw = () => {}, now = () => performance.now(), reducedMotion = null, onNote = null } = {}) {
    this.stage = stage; this.redraw = redraw; this.now = now; this.prefersReduced = reducedMotion; this.onNote = onNote;
    this.calm = false; this.focus = null; this.cell = 0; this.arrows = new Map(); this.dying = []; this.passed = new Set();
    this.hits = []; this.labels = []; this.noteKind = ''; this.drawnAt = -Infinity; this.key = '';
    stage.addAnimator?.(this);
  }
  reduced() {
    if (this.calm) return true;
    if (this.prefersReduced) return !!this.prefersReduced();
    try { return !!(this.motionQuery ??= globalThis.matchMedia?.('(prefers-reduced-motion: reduce)'))?.matches; } catch { return false; }
  }
  /** The window's focus (null: none). `motion`: the update may move (a live update a second or more after the previous
   *  one, or a scrub's transition); a move to another cell always draws its arrows on. Returns whether the overlay needs
   *  a redraw. */
  set(focus, { motion = false } = {}) {
    const now = this.now(), cell = focus?.cell ?? 0, other = cell !== this.cell;
    const moving = !this.reduced() && (motion || other);
    if (other) this.passed = new Set();
    const target = focus?.target && focus.target.every(Number.isFinite) ? focus.target : null;
    const sources = (focus?.sources ?? []).filter(s => s && s.share >= ARROW_MIN_SHARE && Array.isArray(s.from) && s.from.every(Number.isFinite) && target)
      .sort((a, b) => b.share - a.share).slice(0, ARROWS);
    const next = new Map();
    for (const s of sources) {
      const old = other ? null : this.arrows.get(s.label);
      const first = s.share >= ARROW_PULSE_SHARE && !this.passed.has(s.label);
      if (s.share >= ARROW_PULSE_SHARE) this.passed.add(s.label);
      const born = old ? old.born : moving && (first || other) ? now : -Infinity;
      next.set(s.label, { label: s.label, share: s.share, pct: Number.isFinite(s.pct) ? s.pct : null, from: s.from, to: target, name: s.name ?? '', born,
        pulse: old ? old.pulse : moving && first ? now : -Infinity });
    }
    for (const [label, a] of this.arrows) if (!next.has(label) && moving) this.dying.push({ ...a, dyingAt: now });
    this.arrows = next; this.cell = cell; this.focus = focus ?? null; this.hot = focus?.hot ?? 0;
    this.diameter = Number.isFinite(focus?.diameter) && focus.diameter > 0 ? focus.diameter : 10;
    this.missing = !!focus && (focus.sources ?? []).some(s => s && s.share >= ARROW_MIN_SHARE && !(Array.isArray(s.from) && s.from.every(Number.isFinite)) || !target);
    const key = JSON.stringify([cell, target, [...next.values()].map(a => [a.label, Math.round(a.share * 1000), a.from]), this.hot, this.dying.length]);
    const changed = key !== this.key; this.key = key;
    if (changed) this.stage.replan?.();
    return changed;
  }
  clear() { const had = !!this.focus || this.dying.length > 0; this.focus = null; this.cell = 0; this.arrows = new Map(); this.dying = []; this.key = ''; return had; }
  /** Drawn at all: in 2-D, with cells large enough on screen. */
  drawable() { return !this.stage.orbit?.active && this.diameter * (this.stage.scale || 0) >= ARROW_MIN_PX; }
  /** What the window says about the arrows now: '' (drawn, or none to draw), 'zoom' or '3d' (calls onNote on change). */
  sync() {
    const kind = !this.focus || !this.arrows.size ? '' : this.stage.orbit?.active ? '3d' : this.drawable() ? '' : 'zoom';
    if (kind !== this.noteKind) { this.noteKind = kind; try { this.onNote?.(kind); } catch { /* the note is a hint */ } }
    return kind;
  }
  // ---- the stage's animator: frames only while an arrow draws on, pulses or fades ----------------------------------------
  due(now = this.now()) {
    if (!this.drawable()) return null;
    let due = null;
    const consider = t => { if (t != null && (due == null || t < due)) due = t; };
    for (const a of this.arrows.values()) { consider(stepDue(a.born, ARROW_DRAW_MS, this.drawnAt)); consider(stepDue(a.pulse, ARROW_DRAW_MS, this.drawnAt)); }
    for (const a of this.dying) consider(stepDue(a.dyingAt, ARROW_FADE_MS, this.drawnAt));
    return due;
  }
  /** One motion frame in the stage's animation frame (counted apart from the check's own motion frames). */
  frame(now = this.now()) { this.drawnAt = now; this.redraw(); harness?.count('qcDeeperArrowFrames'); return false; }
  /** Draw on the overlay's 2-D context `g` (CSS px). */
  draw(g) {
    const now = this.now(); this.drawnAt = now; this.hits = []; this.labels = [];
    this.sync();
    this.dying = this.dying.filter(a => now - a.dyingAt < ARROW_FADE_MS);
    if ((!this.arrows.size && !this.dying.length) || !this.drawable()) return;
    // Arrows end at the cell's edge (its size as a circle of the same area), so the heads never cover the cell itself.
    const s = this.stage, scale = s.scale || 1, stop = Math.max(4, this.diameter * scale / 2 + 2);
    const labelled = new Set([...this.arrows.values()].sort((x, y) => y.share - x.share).slice(0, ARROW_LABELS).map(a => a.label));
    const items = [...this.dying.map(a => ({ a, fade: 1 - arrowProgress(now - a.dyingAt, ARROW_FADE_MS), dying: true })),
      ...[...this.arrows.values()].map(a => ({ a, fade: 1, dying: false }))];
    g.save(); g.setLineDash([]); g.lineCap = 'round'; g.lineJoin = 'round';
    for (const { a, fade, dying } of items) {
      if (fade <= 0) continue;
      const from = s.toScreen(a.from[0], a.from[1]), to = s.toScreen(a.to[0], a.to[1]);
      const dx = to[0] - from[0], dy = to[1] - from[1], len = Math.hypot(dx, dy);
      if (len < stop + 6) continue;   // the neighbour's centre is (almost) inside the cell on screen: no room for an arrow
      const end = [to[0] - dx / len * stop, to[1] - dy / len * stop], curve = arrowCurve(from, end);
      const t = Number.isFinite(a.born) ? arrowProgress(now - a.born, ARROW_DRAW_MS) : 1;
      const w = arrowWidth(a.share) + (!dying && a.label === this.hot ? 1 : 0);
      g.globalAlpha = fade;
      tracePart(g, curve, t); g.strokeStyle = ARROW_HALO; g.lineWidth = w + 4; g.stroke();
      tracePart(g, curve, t); g.strokeStyle = M.accent; g.lineWidth = w; g.stroke();
      // The head travels with the tip as the arrow draws on.
      const tip = curve.at(t), [ux, uy] = curve.dir(t), size = 3 * w + 4;
      g.beginPath(); g.moveTo(tip[0] + ux * 2, tip[1] + uy * 2);
      g.lineTo(tip[0] - ux * size - uy * size * .55, tip[1] - uy * size + ux * size * .55);
      g.lineTo(tip[0] - ux * size + uy * size * .55, tip[1] - uy * size - ux * size * .55); g.closePath();
      g.fillStyle = ARROW_HALO; g.lineWidth = 3; g.strokeStyle = ARROW_HALO; g.stroke(); g.fillStyle = M.accent; g.fill();
      // One pulse ring at the neighbour, the first time its share passes ARROW_PULSE_SHARE.
      if (Number.isFinite(a.pulse) && now - a.pulse < ARROW_DRAW_MS && !dying) {
        const p = arrowProgress(now - a.pulse, ARROW_DRAW_MS);
        g.globalAlpha = fade * .7 * (1 - p); g.beginPath(); g.arc(from[0], from[1], 5 + 12 * p, 0, Math.PI * 2);
        g.strokeStyle = M.accent; g.lineWidth = 2; g.stroke(); g.globalAlpha = fade;
      }
      if (!dying && t >= 1 && labelled.has(a.label)) {
        // The window row's own percent: rounded down and never above the headline's.
        const text = deep('deepArrowLabel', { p: number(Math.max(1, Number.isFinite(a.pct) ? a.pct : Math.floor(a.share * 100 + 1e-9))) });
        g.font = FONT; g.textBaseline = 'middle'; g.textAlign = 'center';
        const [mx, my] = curve.at(.5), tw = Math.ceil(g.measureText(text).width) + 12;
        pill(g, mx - tw / 2, my - 9, tw, 18); g.fillStyle = 'rgba(14,17,22,.88)'; g.fill();
        g.strokeStyle = M.accent; g.lineWidth = 1; g.stroke();
        g.fillStyle = '#e6eaf0'; g.fillText(text, mx, my + .5);
        this.labels.push(text);
      }
      if (!dying) this.hits.push({ label: a.label, curve, t, w });
    }
    g.restore();
  }
  /** The source whose arrow is under CSS point (x, y), or 0. */
  hit(x, y) {
    for (let i = this.hits.length - 1; i >= 0; i--) {
      const { label, curve, t, w } = this.hits[i], reach = Math.max(ARROW_HIT_PX, w);
      let prev = curve.a;
      for (let k = 1; k <= 16; k++) {
        const p = curve.at(t * k / 16);
        if (segmentDistance(x, y, prev, p) <= reach) return label;
        prev = p;
      }
    }
    return 0;
  }
  /** Every word drawn on the overlay for the arrows now (the harness checks them for banned words). */
  texts() { return [...this.labels]; }
  dispose() { this.stage.removeAnimator?.(this); this.clear(); }
}
function segmentDistance(x, y, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? clamp(((x - a[0]) * dx + (y - a[1]) * dy) / l2, 0, 1) : 0;
  return Math.hypot(x - (a[0] + t * dx), y - (a[1] + t * dy));
}
