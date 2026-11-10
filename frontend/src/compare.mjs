// Compare: a full-screen <dialog id="compare-view">
// over the run page that sets the cells of the run's frame and up to three of the sample's segmentations side by side,
// one column each: its UMAP on top, its mixing below (the number, the like-for-like line at 100 transcripts each, the
// spread of the share from neighbours, the cells in the area, the median transcripts and the share under 100). Every
// column is read on its own count (the run's and a saved proseg result's on proseg's count, Xenium's cells on Xenium's,
// any other segmentation by the transcripts inside its outlines: its head says which), read by the sample's mixing
// model and cut to the one area all of them cover: the engine does all of it (POST and GET /runs/{id}/compare); the
// view shows what the engine says and hands every choice
// (a column added or taken out, Try again) back as a new compare (start). It never sets one column against another: no
// verdict and no direction words (compare-copy.mjs).
//
// While the engine works the view asks for the status (fetchStatus) at most twice a second (STATUS_MS), one ask at a
// time; an ask that fails is made again a little later (RETRY_MS, then twice as long each time), and after FAIL_TRIES in a
// row the view says the engine isn't answering, with Try again. Never while closed or hidden, nor once the engine says
// ready or failed. Columns fill as their numbers come. The UMAP points are downloaded once per column and layout
// (fetchPoints: the engine's little-endian records) and drawn when every column's have come, at one scale for all
// columns. The Map switch (remembered in this browser) shows them on One map (one layout shared by every column: the same
// place is the same kind of cell in every column) or Each its own (the default: each column laid out and grouped alone by
// the engine, turned to line up with the yardstick column's: the same points, in the same order, with their own places
// and clusters, downloaded with them). Colour by Clusters (each its own) or Differences (one map: where a column's cells
// gather against the yardstick's) beside Kind and Mixing.
//
// Painting: a column's points are drawn once into a kept picture (pixels written directly, each dot a small soft disc,
// on a 2-D canvas: no WebGL context); a paint lays that picture on the column's canvas, with the hover's rings. One paint
// per change (a hover, a kind lit from the key, a resize); a new layout or another colouring cross-fades in once, over
// FADE_MS at most 12 paints a second (finite); reduced or calm motion: one paint. A press of the Map switch flies each
// dot from one map to the other over FLY_MS (FLY_FPS paints a second at most), then the view rests. A new compare (a
// column added or taken out) dims the old pictures until its layout comes. Closed or hidden: nothing is scheduled. No CSS
// motion (compare.css).
import { COMPARE_TEXT as T, BIN_LABELS, compare, allTexts } from './compare-copy.mjs';
import { QC_TEXT, number, cellShare, sharePercent, ROUGH_JUDGED, HEADLINE_MIN_JUDGED } from './qc-copy.mjs';
import { MIXING_COLORS, MIXING_CODES } from './stage.js';
import { clusterColorCSS, CLUSTER_PALETTE } from './clusters.js';

export const STATUS_MS = 500;          // the status is asked for at most twice a second while the engine works
export const RETRY_MS = 1000;          // a failed ask made again 1 s later, then 2 s, 4 s, ...
export const FAIL_TRIES = 5;           // ... until this many in a row: then the view says the engine isn't answering
export const MISMATCH_TRIES = 10;      // statuses of another set of columns in a row before the view gives up waiting
export const POINT_TRIES = 3;          // a column's points are asked for at most this often per layout
export const MAX_COLUMNS = 4;          // the run's and three segmentations (1-4 columns, each once)
export const RECORD_BYTES = 10;        // a point: f32 x, f32 y, u8 kind (NO_KIND: none), u8 share (floor(share x 250))
export const NO_KIND = 255;
export const FADE_MS = 300;            // a new layout or colouring cross-fades in over this long ...
export const FADE_FPS = 12;            // ... in at most this many paints a second
export const TICK_MS = Math.ceil(1000 / FADE_FPS);
export const GHOST_ALPHA = 0.3;        // a column's old picture while a new compare works
export const DOT_ALPHA = 0.8;          // a dot's strength (overlapping dots build up)
export const DIM_ALPHA = 0.1;          // the other kinds while one is lit from the key
export const HOVER_PX = 7;             // a dot is named within this many CSS px of the pointer ...
export const GRID_PX = 12;             // ... found through a grid of cells this size (CSS px)
export const PAD_PX = 14;              // the layout keeps this far from a canvas's edges (CSS px)
export const PLOT_PX = Object.freeze([360, 300]);   // a canvas's size until the browser reports it: [w, h] CSS px
export const FIT_LO = 0.01, FIT_HI = 0.99, FIT_MARGIN = 0.08;   // the shared scale: 98 % of the points, and a margin
export const FIT_LIMIT = 8192;         // the scale is read from an even pick of at most this many points
export const SMALL_COVER = 0.1;        // a column covering under a tenth of another's area says so
export const MIN_QV = 20;              // the counting rule's quality when the engine names none (proseg's min-qv default)
// The histogram's bins (BIN_LABELS: 0-2, 2-5, 5-10, 10-20, 20-30, 30-50, 50+ %) in the stage's three steps.
export const BIN_STEPS = Object.freeze(['little', 'little', 'little', 'some', 'some', 'mixed', 'mixed']);
export const MIX_STEPS = Object.freeze(['little', 'some', 'mixed', 'unjudged']);
export const COUNTERS = Object.freeze(['compareOpened', 'compareDraws', 'compareFetches']);
// The engine's words for what made a segmentation (`source`) and for its step (`step`): any
// other step reads "Working…" (the engine's own words never reach the page).
export const SOURCES = Object.freeze({ xenium: 'sourceXenium', imported: 'sourceImported', cellpose: 'sourceCellpose', instanseg: 'sourceInstanseg',
  proseg: 'sourceProseg' });
export const STEPS = Object.freeze({ queued: 'stepQueued', waiting: 'stepQueued', outlines: 'stepOutlines', area: 'stepArea',
  counting: 'stepCounting', scoring: 'stepScoring', mixing: 'stepScoring', kinds: 'stepKinds', umap: 'stepUmap', layout: 'stepUmap' });
// A column's state (engine/compare.py: waiting, counting, working, ready, failed) in words while it has no number yet.
export const COLUMN_STATES = Object.freeze({ waiting: 'columnWaiting', queued: 'columnWaiting', counting: 'columnCounting' });
// Which count a column is read on (the engine's column `count`), in words under its name; a count the engine names that
// the page does not know has none. Proseg's count ("proseg") is named by whose it is (countKeys).
export const COUNTS = Object.freeze({ xenium: 'countXenium', outlines: 'countOutlines' });
/** Each count's label's tooltip: what that count is (its label can be cut short by an ellipsis in a narrow column). */
export const COUNT_TOOLTIPS = Object.freeze({ xenium: 'countXeniumTooltip', outlines: 'countOutlinesTooltip' });
/** A column's count words, [label key, tooltip key] (compare-copy.mjs), or null for a count the page has no words for:
 *  on proseg's count the run's column by its form (H9: "sampled", an iteration counted for a pause, the hold, a finish or
 *  the end, read on the count proseg saves there; else averaged over its random background draw) and any other column
 *  (a table proseg saved in the sample: one draw, as proseg saves it) as that table; Xenium's and the outlines' by
 *  COUNTS. */
export function countKeys(key, count, form = null) {
  if (count === 'proseg') return key !== 'run' ? ['countProsegTable', 'countProsegTableTooltip']
    : form === 'sampled' ? ['countProsegSaved', 'countProsegSavedTooltip'] : ['countProsegAveraged', 'countProsegTooltip'];
  return typeof count === 'string' && Object.hasOwn(COUNTS, count) ? [COUNTS[count], COUNT_TOOLTIPS[count]] : null;
}
/** Whose draw the run column's cut to 100 transcripts starts from (the engine's like_from), in words: proseg's own or
 *  the app's (H6); '' when the engine does not say. */
export const LIKE_FROM = Object.freeze({ proseg: 'likeFromProseg', app: 'likeFromApp' });
export const COLORS = Object.freeze({ background: '#0b0e13', ring: '#ffffff', echo: 'rgba(255,255,255,0.6)', none: '#66717e' });
// Another window started a new comparison of the run (the engine keeps one per run): this view stops and offers Try
// again (its own columns compared anew), never another comparison's numbers or dots under its columns.
export const REPLACED_TEXT = 'Another window started a new comparison of this run.';
// The Map switch, remembered in this browser (MAP_KEY): 'own', each column's own map (the default), or 'one', the map
// every column shares.
export const MAPS = Object.freeze(['own', 'one']);
export const MAP_KEY = 'sopa-companion:compare:map';
export const NO_CLUSTER = 65535;       // a point in no cluster, in a column's own clusters (u16 per point)
export const FLY_MS = 900;             // a press of the Map switch flies the dots over this long ...
export const FLY_FPS = 30;             // ... in at most this many paints a second
export const FLY_TICK_MS = Math.ceil(1000 / FLY_FPS);
// Differences (One map): each dot coloured by its column's share of points in a square of a DIFF_GRID x DIFF_GRID grid
// over the map (smoothed), over the yardstick column's share there, in five steps of log2 of that ratio (DIFF_EDGES:
// about a third or under, a half, the same, twice, three times or over). DIFF_SAME: the yardstick's own dots.
export const DIFF_GRID = 40;
export const DIFF_EDGES = Object.freeze([-1.5, -0.5, 0.5, 1.5]);
export const DIFF_SAME = 2;
export const DIFF_COLORS = Object.freeze(['#4393c3', '#92c5de', '#7a828c', '#f4a582', '#d6604d']);
export const DIFF_WORDS = Object.freeze(['diffThird', 'diffHalf', 'diffSame', 'diffTwice', 'diffThrice']);
const finite = Number.isFinite;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Icons: one stroke weight, the text colour.
const svg = body => `<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
export const ICONS = Object.freeze({
  info: svg('<circle cx="8" cy="8" r="6.25"/><path d="M8 7.25v4M8 4.75v.01"/>'),
  plus: svg('<path d="M8 3.5v9M3.5 8h9"/>'),
  close: svg('<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>'),
  back: svg('<path d="M9.75 3.75L5.5 8l4.25 4.25"/>'),
});

// ---- pure helpers (node tests) ------------------------------------------------------------------------------------
const plain = v => (typeof v === 'string' ? v.trim() : '');
const count = v => (finite(v) && v >= 0 ? Math.round(v) : null);
const box4 = b => (Array.isArray(b) && b.length === 4 && b.every(finite) && b[2] > b[0] && b[3] > b[1] ? b.map(Number) : null);
const hexColor = v => (typeof v === 'string' && /^#[0-9a-f]{6}$/i.test(v.trim()) ? v.trim().toLowerCase() : null);
const rgbOf = hex => { const v = parseInt(hex.slice(1), 16); return [v >> 16, (v >> 8) & 255, v & 255]; };

/** The bytes of `data` (an ArrayBuffer or a view of one, as fetchPoints gives them), else null. */
function bytesOf(data) {
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return null;
}
/** A column's UMAP points (little-endian records of RECORD_BYTES: f32 x, f32 y, u8 kind, u8 share) as
 *  {n, x, y, kind, share}; null when the bytes are not that. A point with a coordinate that is not finite is never drawn. */
export function decodePoints(data) {
  const bytes = bytesOf(data);
  if (!bytes || bytes.byteLength % RECORD_BYTES) return null;
  const n = bytes.byteLength / RECORD_BYTES, view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const x = new Float32Array(n), y = new Float32Array(n), kind = new Uint8Array(n), share = new Uint8Array(n);
  for (let i = 0, o = 0; i < n; i++, o += RECORD_BYTES) {
    x[i] = view.getFloat32(o, true); y[i] = view.getFloat32(o + 4, true); kind[i] = bytes[o + 8]; share[i] = bytes[o + 9];
  }
  return { n, x, y, kind, share };
}

/** A column's own clusters (GET /runs/{id}/compare/clusters/{index}.bin: one little-endian u16 per point, in its points'
 *  order, NO_CLUSTER for none) as a Uint16Array of `n`; null when the bytes are not that. */
export function decodeClusters(data, n) {
  const bytes = bytesOf(data);
  if (!bytes || !Number.isInteger(n) || n < 0 || bytes.byteLength !== 2 * n) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), out = new Uint16Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getUint16(2 * i, true);
  return out;
}
/** Each cluster's colour slot (its id under 255; NO_CLUSTER and any later id 255: the palette's grey). */
export function clusterSlots(ids) { return Uint8Array.from(ids, id => (id < 255 ? id : 255)); }

/** One column of the status checked ({key, name, state, count, form, likeFrom, cells, tiles, judged, mixed, like100,
 *  likeCells, histogram, medianTx, under100, points, sampled, own, clusters, matched, error}); null without a key. A histogram is kept only as 7
 *  counts; `tiles`: the 100 µm squares the column covers by itself (the engine's, once it has found the area; null
 *  before); `count`: which count it is read on ("proseg" | "xenium" | "outlines"), null when the engine names none the
 *  page knows; `form`: the run column's count, "sampled" (the count proseg saves at its iteration) or "averaged", null
 *  when not said; `likeFrom`: whose draw its cut to 100 starts from ("proseg" | "app"), null when not said;
 *  `likeCells`: the cells the like-for-like line is over (those whose whole transcripts reach 100: the run's column cuts
 *  its sampled count, or a draw of it, where a few judged on the averaged count fall short), null when the engine does
 *  not say. */
function normaliseColumn(c) {
  if (!c || typeof c !== 'object' || typeof c.key !== 'string' || !c.key) return null;
  const hist = Array.isArray(c.histogram) && c.histogram.length === BIN_LABELS.length && c.histogram.every(v => finite(v) && v >= 0)
    ? c.histogram.map(v => Math.round(v)) : null;
  return { key: c.key, name: plain(c.name), state: typeof c.state === 'string' ? c.state : 'working',
    count: typeof c.count === 'string' && (c.count === 'proseg' || Object.hasOwn(COUNTS, c.count)) ? c.count : null,
    form: ['sampled', 'averaged'].includes(c.form) ? c.form : null, likeFrom: Object.hasOwn(LIKE_FROM, c.like_from ?? '') ? c.like_from : null,
    cells: count(c.cells_in_area), tiles: count(c.tiles),
    judged: count(c.judged), mixed: finite(c.mixed_per100) ? c.mixed_per100 : null, like100: finite(c.like100_per100) ? c.like100_per100 : null,
    likeCells: count(c.like100_cells), histogram: hist,
    medianTx: finite(c.median_tx) ? c.median_tx : null, under100: finite(c.under100_share) ? clamp(c.under100_share, 0, 1) : null,
    points: count(c.umap_points) ?? 0, sampled: c.sampled === true, own: c.own === true, clusters: count(c.clusters) ?? 0,
    matched: count(c.clusters_matched) ?? 0, error: plain(c.error) };
}
/** A view box from the engine ([x0, y0, x1, y1], or {x: [x0, x1], y: [y0, y1]}) checked; null when it is not one. */
const boundsOf = b => box4(Array.isArray(b) ? b : b && Array.isArray(b.x) && Array.isArray(b.y) ? [b.x[0], b.y[0], b.x[1], b.y[1]] : null);
/** The counting rule of every column (the status's `rule`: the run's own input filters) as {minQv (null: not given),
 *  exclude (the gene-name pattern the run left out, null for proseg's default)}; null without one. */
function normaliseRule(r) {
  if (!r || typeof r !== 'object' || Array.isArray(r)) return null;
  return { minQv: finite(r.min_qv) && r.min_qv >= 0 ? r.min_qv : null, exclude: plain(r.exclude) || null };
}
/** The status (GET /runs/{id}/compare) checked and normalised: {state, step,
 *  progress {done, of} | null, model, area {box, size_um, tiles} | null, kinds [{index, name, color}] | null, columns
 *  [...], layout (how many layouts the points carried so far: they are fetched again when it grows), job (which
 *  comparison: the engine's id, null from an engine that names none), seq (the run's frame compared, or null), bounds ([x0, y0,
 *  x1, y1]: the layout's 2nd-98th percentiles, or null), error (why a whole compare failed), kindsReason (why no kinds:
 *  needs_detail, no_source, failed, busy), layoutError (too_few_cells, or why there is no picture), rule (the counting
 *  rule: {minQv, exclude} | null), ownLayout (the engine's word for each column's own map, null without them), ownBounds
 *  (their shared view box, as bounds) and ownError (why there are none)}; null when it is not one. Lenient about extra
 *  fields; a column without a key, or a second one of a key, is left out; a kind without a colour of its own takes the
 *  clusters' colour of its index. */
export function normaliseStatus(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const state = ['working', 'ready', 'failed'].includes(raw.state) ? raw.state : 'working';
  const p = raw.progress;
  const progress = p && finite(p.done) && finite(p.of) && p.of > 0 ? { done: clamp(Math.round(p.done), 0, Math.round(p.of)), of: Math.round(p.of) } : null;
  const a = raw.area && typeof raw.area === 'object' ? raw.area : null, box = box4(a?.box);
  const size = Array.isArray(a?.size_um) && a.size_um.length === 2 && a.size_um.every(v => finite(v) && v > 0) ? a.size_um.map(Number)
    : box ? [box[2] - box[0], box[3] - box[1]] : null;
  const area = size ? { box, size_um: size, tiles: count(a?.tiles) } : null;
  const kinds = Array.isArray(raw.kinds) ? raw.kinds.filter(k => k && Number.isInteger(k.index) && k.index >= 0 && k.index < NO_KIND)
    .map(k => ({ index: k.index, name: plain(k.name), color: hexColor(k.color) ?? clusterColorCSS(k.index) })) : null;
  const seen = new Set(), columns = [];
  for (const c of Array.isArray(raw.columns) ? raw.columns : []) {
    const col = normaliseColumn(c);
    if (col && !seen.has(col.key)) { seen.add(col.key); columns.push(col); }
  }
  const bounds = boundsOf(raw.bounds);
  return { state, step: plain(raw.step) || null, progress, model: raw.model == null ? null : String(raw.model), area, kinds, columns,
    layout: count(raw.layout), job: plain(raw.job) || null, seq: Number.isInteger(raw.seq) && raw.seq >= 0 ? raw.seq : null,
    bounds, error: plain(raw.error) || null, kindsReason: plain(raw.kinds_reason) || null,
    layoutError: plain(raw.layout_error) || null, rule: normaliseRule(raw.rule), ownLayout: plain(raw.own_layout) || null,
    ownBounds: boundsOf(raw.own_bounds), ownError: plain(raw.own_error) || null };
}

/** The sample's segmentations (GET /samples/{id}/segmentations: {segmentations: [...]} or the list itself)
 *  as [{key, name, source, model, count, extent, table}]; null when it is not one. */
export function normaliseList(raw) {
  const list = Array.isArray(raw) ? raw : Array.isArray(raw?.segmentations) ? raw.segmentations : null;
  if (!list) return null;
  const seen = new Set();
  return list.filter(s => s && typeof s.key === 'string' && s.key && s.key !== 'run' && !seen.has(s.key) && seen.add(s.key))
    .map(s => ({ key: s.key, name: plain(s.name) || s.key, source: typeof s.source === 'string' ? s.source : 'other', model: plain(s.model) || null,
      count: count(s.count), extent: box4(s.extent), table: typeof s.table === 'string' && s.table ? s.table : null }));
}
/** The columns a first Compare opens with: the run, Xenium's cells (one with its table first: never the nuclei
 *  when the cells are there) and the earlier imported segmentation, when the sample has them. */
export function defaultColumns(items) {
  const list = Array.isArray(items) ? items : [];
  const xenium = list.find(s => s.source === 'xenium' && s.table) ?? list.find(s => s.source === 'xenium' && !/nucle/i.test(s.key)) ?? null;
  const imported = list.find(s => s.source === 'imported' && s !== xenium) ?? null;
  return ['run', ...[xenium, imported].filter(Boolean).map(s => s.key)];
}

/** One scale for every column's points ([x0, y0, x1, y1] in the layout's units): the 1st to 99th percentile of x and of
 *  y over all of them (an even pick of at most `limit`), widened by FIT_MARGIN each side; null without a finite point. */
export function fitLayout(sets, limit = FIT_LIMIT) {
  const total = sets.reduce((a, s) => a + (s?.n ?? 0), 0);
  if (!total) return null;
  const step = Math.max(1, total / limit), xs = [], ys = [];
  let next = 0, i = 0;
  for (const s of sets) {
    for (let k = 0; k < (s?.n ?? 0); k++, i++) {
      if (i < next) continue;
      next += step;
      const x = s.x[k], y = s.y[k];
      if (finite(x) && finite(y)) { xs.push(x); ys.push(y); }
    }
  }
  if (!xs.length) return null;
  xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
  const q = (a, f) => a[clamp(Math.round(f * (a.length - 1)), 0, a.length - 1)];
  const x0 = q(xs, FIT_LO), x1 = q(xs, FIT_HI), y0 = q(ys, FIT_LO), y1 = q(ys, FIT_HI);
  const w = Math.max(x1 - x0, 1e-6), h = Math.max(y1 - y0, 1e-6);
  return [x0 - w * FIT_MARGIN, y0 - h * FIT_MARGIN, x1 + w * FIT_MARGIN, y1 + h * FIT_MARGIN];
}
/** A view box widened by FIT_MARGIN each side. */
export const widen = b => { const w = b[2] - b[0], h = b[3] - b[1]; return [b[0] - w * FIT_MARGIN, b[1] - h * FIT_MARGIN, b[2] + w * FIT_MARGIN, b[3] + h * FIT_MARGIN]; };
/** The square of (x, y) on a g x g grid over `bounds` (a place beyond them in the edge squares); -1 for one that is
 *  not finite. */
export function gridCell(x, y, bounds, g) {
  if (!finite(x) || !finite(y)) return -1;
  const gx = clamp(Math.floor(((x - bounds[0]) / (bounds[2] - bounds[0])) * g), 0, g - 1);
  const gy = clamp(Math.floor(((y - bounds[1]) / (bounds[3] - bounds[1])) * g), 0, g - 1);
  return gy * g + gx;
}
/** [1 2 1] / 4 across, then down (an edge square keeps its outer quarter): the sum stays. */
function smooth(grid, g) {
  const across = new Float64Array(g * g), out = new Float64Array(g * g);
  for (let y = 0; y < g; y++) {
    for (let x = 0; x < g; x++) {
      const o = y * g;
      across[o + x] = (grid[o + Math.max(0, x - 1)] + 2 * grid[o + x] + grid[o + Math.min(g - 1, x + 1)]) / 4;
    }
  }
  for (let y = 0; y < g; y++) {
    for (let x = 0; x < g; x++) out[y * g + x] = (across[Math.max(0, y - 1) * g + x] + 2 * across[y * g + x] + across[Math.min(g - 1, y + 1) * g + x]) / 4;
  }
  return out;
}
/** A column's points as shares of a g x g grid over `bounds`, smoothed twice (about a square either side), summing to
 *  1; null without a finite point. */
export function shareGrid(points, bounds, g = DIFF_GRID) {
  let grid = new Float64Array(g * g), n = 0;
  for (let i = 0; i < points.n; i++) { const c = gridCell(points.x[i], points.y[i], bounds, g); if (c >= 0) { grid[c]++; n++; } }
  if (!n) return null;
  grid = smooth(smooth(grid, g), g);
  const total = grid.reduce((a, v) => a + v, 0);
  for (let c = 0; c < grid.length; c++) grid[c] /= total;
  return grid;
}
/** The step (0-4) of log2 of a column's share over the yardstick's (DIFF_EDGES); DIFF_SAME for a value that is not
 *  finite. */
export function diffStep(log2) {
  if (!finite(log2)) return DIFF_SAME;
  let step = 0;
  for (const edge of DIFF_EDGES) if (log2 >= edge) step++;
  return step;
}
/** Differences on the one map: for each set of points (null for none) {steps (0-4 per point), ratio (its column's share
 *  of points around it over the yardstick's, `sets[ref]`, there)}; the yardstick's own points DIFF_SAME, ratio 1. A
 *  floor of a tenth of an even share keeps a square where either has next to nothing from reading as a vast ratio.
 *  Every set null when the yardstick has no points. */
export function differences(sets, ref, bounds, g = DIFF_GRID) {
  const base = sets[ref] ? shareGrid(sets[ref], bounds, g) : null, floor = 0.1 / (g * g);
  return sets.map((set, c) => {
    const grid = base && set ? (c === ref ? base : shareGrid(set, bounds, g)) : null;
    if (!grid) return null;
    const steps = new Uint8Array(set.n), ratio = new Float32Array(set.n);
    for (let i = 0; i < set.n; i++) {
      const cell = gridCell(set.x[i], set.y[i], bounds, g), r = c === ref || cell < 0 ? 1 : (grid[cell] + floor) / (base[cell] + floor);
      ratio[i] = r; steps[i] = c === ref ? DIFF_SAME : diffStep(Math.log2(r));
    }
    return { steps, ratio };
  });
}
/** Smooth start and landing (0-1). */
export const ease = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
/** Draws a flight's dots where they are (FLY_PLACE: device px as they come, y negated as flyPoints gives it). */
export const FLY_PLACE = Object.freeze({ ox: 0, oy: 0, k: 1 });
/** A flight's dots `t` (0-1) of the way from where points `a` lie (placed by `pa`) to where `b` lie (placed by `pb`):
 *  {n, x, y} in device px, y negated (FLY_PLACE draws them there); into `out` when it holds as many. */
export function flyPoints(a, pa, b, pb, t, out = null) {
  const n = a.n, o = out && out.x.length === n ? out : { n, x: new Float32Array(n), y: new Float32Array(n) };
  for (let i = 0; i < n; i++) {
    const ax = pa.ox + pa.k * a.x[i], ay = pa.oy - pa.k * a.y[i], bx = pb.ox + pb.k * b.x[i], by = pb.oy - pb.k * b.y[i];
    o.x[i] = ax + (bx - ax) * t; o.y[i] = -(ay + (by - ay) * t);
  }
  o.n = n;
  return o;
}
/** Where the layout's `bounds` land on a canvas of w x h device px keeping `pad` px clear, one scale for x and y and y
 *  up (as the run's UMAP draws it): a point is at (ox + k x, oy - k y). */
export function placement(bounds, w, h, pad = 0) {
  const bw = bounds[2] - bounds[0], bh = bounds[3] - bounds[1];
  const k = Math.max(1e-9, Math.min(Math.max(1, w - 2 * pad) / bw, Math.max(1, h - 2 * pad) / bh));
  return { k, ox: w / 2 - (k * (bounds[0] + bounds[2])) / 2, oy: h / 2 + (k * (bounds[1] + bounds[3])) / 2, w, h };
}
/** A dot's radius in CSS px: smaller the more points share the canvas (`most`: the most in one column, so every column's
 *  dots are the same size), between 1.1 and 2.4. */
export function dotRadius(most, cssW, cssH) {
  return clamp(0.21 * Math.sqrt((Math.max(1, cssW) * Math.max(1, cssH)) / Math.max(1, most)), 1.1, 2.4);
}

const SUB = 4;   // a dot's place is kept to a quarter of a device pixel
const STAMPS = new Map();
/** A soft disc of radius `r` device px for each quarter-pixel place inside a pixel (SUB x SUB of them): flat [dx, dy,
 *  coverage, ...] around the pixel holding the dot's middle. */
export function stamps(r) {
  const key = Math.round(r * 100);
  let out = STAMPS.get(key);
  if (out) return out;
  out = [];
  const R = Math.ceil(r + 1);
  for (let fy = 0; fy < SUB; fy++) {
    for (let fx = 0; fx < SUB; fx++) {
      const cx = (fx + 0.5) / SUB, cy = (fy + 0.5) / SUB, list = [];
      for (let dy = -R; dy <= R; dy++) {
        for (let dx = -R; dx <= R; dx++) {
          const c = clamp(r + 0.5 - Math.hypot(dx + 0.5 - cx, dy + 0.5 - cy), 0, 1);
          if (c > 0.02) list.push(dx, dy, c);
        }
      }
      out.push(Float32Array.from(list));
    }
  }
  STAMPS.set(key, out);
  return out;
}
/** Draws `points` into `pixels` (RGBA, w x h, row-major, opaque) placed by `place`: each a soft disc of radius `r` device
 *  px in the colour of its slot (`slots[i]`, an index into `palette`, flat [r, g, b, ...]) at `alpha`, laid over what is
 *  there. With `lit` (a flag per point: the kind lit from the key) the others go first at `dim`, the lit ones over them.
 *  Returns how many dots landed on the canvas. */
export function rasterise(pixels, w, h, points, place, slots, palette, { r = 1.2, alpha = DOT_ALPHA, lit = null, dim = DIM_ALPHA } = {}) {
  const st = stamps(r), reach = Math.ceil(r + 1);
  let drawn = 0;
  const pass = (only, a) => {
    for (let i = 0; i < points.n; i++) {
      if (only != null && lit[i] !== only) continue;
      const px = place.ox + place.k * points.x[i], py = place.oy - place.k * points.y[i];
      if (!(px > -reach && py > -reach && px < w + reach && py < h + reach)) continue;   // off the canvas (NaN too)
      const ix = Math.floor(px), iy = Math.floor(py);
      const s = st[Math.min(SUB - 1, Math.floor((py - iy) * SUB)) * SUB + Math.min(SUB - 1, Math.floor((px - ix) * SUB))];
      const c = slots[i] * 3, R = palette[c], G = palette[c + 1], B = palette[c + 2];
      for (let e = 0; e < s.length; e += 3) {
        const x = ix + s[e], y = iy + s[e + 1];
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        const o = (y * w + x) * 4, t = a * s[e + 2];
        pixels[o] += (R - pixels[o]) * t; pixels[o + 1] += (G - pixels[o + 1]) * t; pixels[o + 2] += (B - pixels[o + 2]) * t;
      }
      drawn++;
    }
  };
  if (lit) { pass(0, dim); pass(1, alpha); } else pass(null, alpha);
  return drawn;
}
/** A cell's step on the stage's ramp from its share code: 0 little (under 10 %), 1 some (10-30 %), 2 mixed (30 % or
 *  above), 3 not judged. */
export function stepOf(code) {
  return !(code < MIXING_CODES.unjudged) ? 3 : code >= MIXING_CODES.mixed ? 2 : code >= MIXING_CODES.some ? 1 : 0;
}
/** The palettes, flat [r, g, b] per slot: by kind (slot = the kind's index; NO_KIND and kinds not listed grey), by
 *  mixing (MIX_STEPS: the stage's colours), by cluster (slot = its id, the clusters' palette; 255 grey) and by
 *  difference (DIFF_COLORS: the five steps). */
export function palettes(kinds) {
  const kind = new Uint8Array(256 * 3), none = rgbOf(COLORS.none);
  for (let k = 0; k < 256; k++) kind.set(none, k * 3);
  for (const k of kinds ?? []) kind.set(rgbOf(k.color), k.index * 3);
  const mixing = Uint8Array.from(MIX_STEPS.flatMap(s => rgbOf(MIXING_COLORS[s])));
  const clusters = new Uint8Array(256 * 3);
  for (let k = 0; k < 256; k++) clusters.set(k < 255 ? rgbOf(CLUSTER_PALETTE[k % CLUSTER_PALETTE.length]) : none, k * 3);
  return { kind, mixing, clusters, diff: Uint8Array.from(DIFF_COLORS.flatMap(rgbOf)) };
}
/** A grid of the points on a canvas (cells of `cell` device px) for finding the one under the pointer. */
export function pointGrid(points, place, cell) {
  const cols = Math.max(1, Math.ceil(place.w / cell)), rows = Math.max(1, Math.ceil(place.h / cell));
  const head = new Int32Array(cols * rows).fill(-1), next = new Int32Array(points.n).fill(-1);
  for (let i = 0; i < points.n; i++) {
    const gx = Math.floor((place.ox + place.k * points.x[i]) / cell), gy = Math.floor((place.oy - place.k * points.y[i]) / cell);
    if (!(gx >= 0 && gy >= 0 && gx < cols && gy < rows)) continue;
    next[i] = head[gy * cols + gx]; head[gy * cols + gx] = i;
  }
  return { cols, rows, cell, head, next };
}
/** The point nearest (x, y) (device px) within `reach`, or null. The one drawn last wins a tie (it is on top). */
export function nearestPoint(points, place, grid, x, y, reach) {
  let best = null, bestD = reach * reach;
  const g0 = Math.floor((x - reach) / grid.cell), g1 = Math.floor((x + reach) / grid.cell);
  const h0 = Math.floor((y - reach) / grid.cell), h1 = Math.floor((y + reach) / grid.cell);
  for (let gy = Math.max(0, h0); gy <= Math.min(grid.rows - 1, h1); gy++) {
    for (let gx = Math.max(0, g0); gx <= Math.min(grid.cols - 1, g1); gx++) {
      for (let i = grid.head[gy * grid.cols + gx]; i >= 0; i = grid.next[i]) {
        const dx = place.ox + place.k * points.x[i] - x, dy = place.oy - place.k * points.y[i] - y, d = dx * dx + dy * dy;
        if (d < bestD || (d === bestD && best != null && i > best)) { bestD = d; best = i; }
      }
    }
  }
  return best;
}
/** A fade's progress `elapsed` ms after it began: the first paint already moves one tick, the end at FADE_MS. */
export function fadeProgress(elapsed) { return clamp((elapsed + TICK_MS) / (FADE_MS + TICK_MS), 0, 1); }
/** The area's size in words: "2.1 × 1.4 mm" from 1 mm on, else "640 × 410 µm". */
export function sizeText(size) {
  if (!Array.isArray(size) || !size.every(finite)) return T.missing;
  const [w, h] = size;
  return Math.max(w, h) >= 1000 ? compare('mm', { w: number(w / 1000, 1), h: number(h / 1000, 1) }) : compare('um', { w: number(w), h: number(h) });
}
/** Each bin's share of the counts (0-1); null without counts. */
export function binShares(hist) {
  const total = Array.isArray(hist) ? hist.reduce((a, v) => a + v, 0) : 0;
  return total > 0 ? hist.map(v => v / total) : null;
}
/** A number shown whole (floor(x + 0.5), as the mixing check rounds its headline). */
const whole = v => number(Math.floor(v + 0.5));
const area4 = b => (b ? Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]) : null);
/** Box `b` cut to box `limit` (none: as it is); an empty cut has no area. */
const within = (b, limit) => (b && limit ? [Math.max(b[0], limit[0]), Math.max(b[1], limit[1]), Math.min(b[2], limit[2]), Math.min(b[3], limit[3])] : b);

// ---- the view -----------------------------------------------------------------------------------------------------------
/**
 * new CompareView(host, options): the view inside `host` (the page's body), closed until open(). Options (main.js gives
 * them from api.js): fetchStatus(runId) -> the status (GET /runs/{id}/compare), fetchPoints(runId, index, job, layout)
 * -> the bytes of column `index` (its place in the status's columns, from 0: GET /runs/{id}/compare/points/{index}.bin
 * of comparison `job` at layout `layout`, an ArrayBuffer; a 409 when the engine's comparison is another one now; a fifth
 * argument 'own' asks for its own map's points (?own=1), 'clusters' for its own clusters), start(runId, columns) -> POST /runs/{id}/compare {columns} (starts or reuses), listSegmentations(sampleId)
 * -> the sample's segmentations, reducedMotion and calmMotion (booleans or functions), hook (the harness:
 * COUNTERS); optional learnDetails(sampleId) (POST /samples/{id}/qc {detail: true}: "Learn the details" when the model has
 * no kinds) and onClose() (Back, Esc: the page underneath carries on); storage (the Map switch is remembered there:
 * localStorage unless given, null for none). Testing: clock {now, setTimeout, clearTimeout, requestAnimationFrame,
 * cancelAnimationFrame}.
 */
export class CompareView {
  constructor(host, options = {}) {
    this.host = host; this.o = options; this.doc = host?.ownerDocument ?? globalThis.document;
    this.hook = options.hook ?? null;
    this.clock = options.clock ?? {
      now: () => globalThis.performance?.now() ?? Date.now(),
      setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms), clearTimeout: id => globalThis.clearTimeout(id),
      requestAnimationFrame: fn => globalThis.requestAnimationFrame(fn), cancelAnimationFrame: id => globalThis.cancelAnimationFrame(id),
    };
    this.listeners = []; this.timers = new Map(); this.paintFrame = null; this.observer = null;
    this.isOpen = false; this.state = 'closed'; this.runId = null; this.sampleId = null; this.iteration = null; this.runBox = null; this.seq = null;
    this.kept = [];   // the constants the run's kind keeps from before proseg's count (open's `kept`)
    this.chosen = new Map();   // the columns last chosen per sample (while the page lives)
    this.colourBy = null;      // a press's choice ('kind' | 'mixing' | 'groups': Clusters or Differences, by the map); null:
                               // by kind when the model has kinds, else by mixing
    try { this.storage = options.storage === undefined ? globalThis.localStorage ?? null : options.storage; } catch { this.storage = null; }
    this.mapMode = this.readMap();   // the Map switch: 'own' (each its own) or 'one' (one map)
    this.dpr = 1; this.asking = false; this.paintedAt = -Infinity; this.modal = false; this.note = '';
    this.draws = 0; this.fetches = 0; this.polls = 0; this.downloads = 0; this.opened = 0; this.gen = 0; this.listGen = 0;
    this.cols = new Map();
    this.clear();
    this.build();
    for (const name of COUNTERS) this.hook?.count(name, 0);
    this.listen(this.doc, 'visibilitychange', () => this.visibility());
  }

  // ---- helpers ------------------------------------------------------------------------------------------------------
  el(tag, className = '', text = '') { const n = this.doc.createElement(tag); if (className) n.className = className; if (text) n.textContent = text; return n; }
  listen(node, type, fn, opts) { node?.addEventListener?.(type, fn, opts); this.listeners.push([node, type, fn, opts]); }
  call(name, ...args) { try { return this.o[name]?.(...args); } catch { return undefined; } }
  later(key, ms, fn) { this.cancel(key); this.timers.set(key, this.clock.setTimeout(() => { this.timers.delete(key); fn(); }, Math.max(0, ms))); }
  cancel(key) { if (this.timers.has(key)) { this.clock.clearTimeout(this.timers.get(key)); this.timers.delete(key); } }
  now() { return this.clock.now(); }
  get hidden() { return !!this.doc?.hidden; }
  /** Reduced or calm motion: no cross-fades, one paint per change. */
  get still() {
    const flag = v => { try { return typeof v === 'function' ? !!v() : !!v; } catch { return false; } };
    if (flag(this.o.calmMotion)) return true;
    if (this.o.reducedMotion !== undefined) return flag(this.o.reducedMotion);
    try { return !!(this.media ??= globalThis.matchMedia?.('(prefers-reduced-motion: reduce)'))?.matches; } catch { return false; }
  }
  /** The Map switch as this browser last left it ('own' without one). */
  readMap() { try { const v = this.storage?.getItem?.(MAP_KEY); return MAPS.includes(v) ? v : 'own'; } catch { return 'own'; } }
  saveMap() { try { this.storage?.setItem?.(MAP_KEY, this.mapMode); } catch { /* a convenience */ } }
  /** A request to the engine (status, points, start): counted for the harness. */
  counted() { this.fetches++; this.hook?.count('compareFetches'); }
  /** A button of the view: never disabled; a press that cannot do its work says why. */
  button(text, action, className = 'cmp-button', icon = '') {
    const b = this.el('button', className); b.type = 'button'; b.dataset.action = action;
    if (icon) b.innerHTML = icon;
    if (text) b.append(this.el('span', '', text));
    return b;
  }
  /** An ⓘ that explains the line it sits on (its tooltip; focusable for the keyboard). */
  info() {
    const i = this.el('span', 'cmp-info'); i.tabIndex = 0; i.setAttribute('role', 'img'); i.innerHTML = ICONS.info;
    return i;
  }

  // ---- the DOM, built once ------------------------------------------------------------------------------------------
  build() {
    const el = (t, c, x) => this.el(t, c, x);
    const root = this.root = el('dialog', 'cmp');
    root.id = 'compare-view'; root.tabIndex = -1; root.setAttribute('aria-labelledby', 'cmp-title');
    root.dataset.qcCopy = ''; root.dataset.state = 'closed';   // the harness's copy check and its state
    // Header: the title; Add (its menu of the sample's segmentations), Map One map | Each its own, Colour by Kind | Mixing |
    // Clusters (each its own) or Differences (one map), Back.
    const head = el('header', 'cmp-head');
    const names = el('div', 'cmp-names');
    const title = el('h2', 'cmp-title', T.title); title.id = 'cmp-title';
    names.append(title, el('p', 'cmp-sub', T.sub));
    const tools = el('div', 'cmp-tools');
    const addWrap = el('div', 'cmp-add-wrap');
    this.addButton = this.button(T.add, 'add', 'cmp-button cmp-add', ICONS.plus); this.addButton.title = T.addTooltip;
    this.addButton.setAttribute('aria-haspopup', 'menu'); this.addButton.setAttribute('aria-expanded', 'false');
    const menu = this.menu = el('div', 'cmp-menu'); menu.setAttribute('role', 'menu'); menu.hidden = true;
    this.menuItems = el('div', 'cmp-menu-items');
    this.menuNote = el('p', 'cmp-menu-note'); this.menuNote.hidden = true;
    this.menuRetry = this.button(T.tryAgain, 'list-retry', 'cmp-button cmp-text'); this.menuRetry.hidden = true;
    menu.append(el('p', 'cmp-menu-title', T.addTitle), this.menuItems, this.menuNote, this.menuRetry);
    addWrap.append(this.addButton, menu);
    const map = el('div', 'cmp-colour cmp-map'); map.setAttribute('role', 'group'); map.setAttribute('aria-label', T.mapBy);
    const mapSeg = el('div', 'cmp-seg');
    this.oneButton = this.button(T.mapOne, 'map-one', 'cmp-seg-button'); this.oneButton.title = T.mapOneTooltip;
    this.ownButton = this.button(T.mapOwn, 'map-own', 'cmp-seg-button');
    this.oneButton.dataset.map = 'one'; this.ownButton.dataset.map = 'own';
    mapSeg.append(this.oneButton, this.ownButton);
    map.append(el('span', 'cmp-label', T.mapBy), mapSeg);
    const colour = el('div', 'cmp-colour'); colour.setAttribute('role', 'group'); colour.setAttribute('aria-label', T.colourBy);
    const seg = el('div', 'cmp-seg');
    this.kindButton = this.button(T.byKind, 'kind', 'cmp-seg-button'); this.kindButton.title = T.byKindTooltip;
    this.mixingButton = this.button(T.byMixing, 'mixing', 'cmp-seg-button'); this.mixingButton.title = T.byMixingTooltip;
    this.kindButton.dataset.colourBy = 'kind'; this.mixingButton.dataset.colourBy = 'mixing';   // the harness's toggle_js
    // The third colouring follows the map: Clusters on each its own, Differences on one map (renderColour).
    this.groupButton = this.button('', 'groups', 'cmp-seg-button'); this.groupLabel = el('span', '', T.byClusters);
    this.groupButton.append(this.groupLabel);
    seg.append(this.kindButton, this.mixingButton, this.groupButton);
    colour.append(el('span', 'cmp-label', T.colourBy), seg);
    this.backButton = this.button(T.back, 'back', 'cmp-button cmp-back', ICONS.back); this.backButton.title = T.backTooltip;
    tools.append(addWrap, map, colour, this.backButton);
    head.append(names, tools);
    // The line under it: the area all of them cover (its ⓘ: the squares and how cells count), how far the engine is.
    const bar = el('div', 'cmp-bar');
    this.areaNode = el('p', 'cmp-area'); this.areaNode.dataset.compareArea = '';
    this.areaText = el('span', 'cmp-area-text'); this.areaInfo = this.info(); this.areaInfo.hidden = true;
    this.areaNode.append(this.areaText, this.areaInfo);
    this.progressNode = el('p', 'cmp-progress'); this.progressNode.setAttribute('aria-live', 'polite');
    // What could not be done (the engine's failure, an engine that isn't answering, a press that explains), Try again: in
    // the same line, so nothing under it moves (the plots keep their size).
    this.alert = el('div', 'cmp-alert'); this.alert.setAttribute('role', 'alert'); this.alert.hidden = true;
    this.alertText = el('p', 'cmp-alert-text');
    this.retryButton = this.button(T.tryAgain, 'retry', 'cmp-button cmp-text'); this.retryButton.hidden = true;
    this.alert.append(this.alertText, this.retryButton);
    bar.append(this.areaNode, this.progressNode, this.alert);
    this.grid = el('div', 'cmp-columns'); this.grid.dataset.count = '0';
    // The key under the columns: the kinds (swatches named by their genes) or the mixing's three steps; while the model
    // has no kinds, why, and Learn the details.
    const key = this.keyBox = el('footer', 'cmp-key'); key.dataset.compareKey = '';
    this.kindsKey = el('div', 'cmp-key-group'); this.kindsKey.dataset.by = 'kind';
    const kindsHead = el('span', 'cmp-key-title', T.kindsTitle), kindsInfo = this.info(); kindsInfo.title = T.kindsTooltip; kindsInfo.setAttribute('aria-label', T.kindsTooltip);
    kindsHead.append(kindsInfo);
    this.kindsList = el('ul', 'cmp-key-items');
    this.kindsKey.append(kindsHead, this.kindsList);
    this.mixKey = el('div', 'cmp-key-group'); this.mixKey.dataset.by = 'mixing';
    const mixList = el('ul', 'cmp-key-items');
    [T.mixLittle, T.mixSome, T.mixMixed].forEach((text, step) => mixList.append(this.keyItem('mixing', step, MIXING_COLORS[MIX_STEPS[step]], text)));
    this.mixKey.append(el('span', 'cmp-key-title', T.mixingTitle), mixList);
    // Clusters: the numbers of the groups on screen (each column's own); Differences: the five steps.
    this.clustersKey = el('div', 'cmp-key-group'); this.clustersKey.dataset.by = 'clusters'; this.clustersKey.hidden = true;
    const clustersHead = el('span', 'cmp-key-title', T.clustersTitle); this.clustersInfo = this.info(); clustersHead.append(this.clustersInfo);
    this.clustersList = el('ul', 'cmp-key-items');
    this.clustersKey.append(clustersHead, this.clustersList);
    this.diffKey = el('div', 'cmp-key-group'); this.diffKey.dataset.by = 'differences'; this.diffKey.hidden = true;
    const diffHead = el('span', 'cmp-key-title', T.diffTitle); this.diffInfo = this.info(); diffHead.append(this.diffInfo);
    const diffList = el('ul', 'cmp-key-items');
    DIFF_WORDS.forEach((word, step) => diffList.append(this.keyItem('differences', step, DIFF_COLORS[step], T[word])));
    this.diffKey.append(diffHead, diffList);
    this.keyNote = el('p', 'cmp-key-note'); this.keyNote.hidden = true;
    this.keyNoteText = el('span', '');
    this.learnButton = this.button(T.learn, 'learn', 'cmp-button cmp-text'); this.learnButton.title = T.learnTooltip; this.learnButton.hidden = true;
    this.keyNote.append(this.keyNoteText, this.learnButton);
    key.append(this.kindsKey, this.mixKey, this.clustersKey, this.diffKey, this.keyNote);
    root.append(head, bar, this.grid, key);
    this.host?.append?.(root);
    // ---- behaviour
    this.addButton.onclick = event => { event?.stopPropagation?.(); this.toggleMenu(); };
    this.menuRetry.onclick = event => { event?.stopPropagation?.(); this.loadList(false); };
    this.kindButton.onclick = () => this.setColour('kind');
    this.mixingButton.onclick = () => this.setColour('mixing');
    this.groupButton.onclick = () => this.setColour('groups');
    this.oneButton.onclick = () => this.setMap('one');
    this.ownButton.onclick = () => this.setMap('own');
    this.backButton.onclick = () => this.close();
    this.retryButton.onclick = () => { if (this.isOpen) this.begin(this.keys); };
    this.learnButton.onclick = () => this.learn();
    // Esc: the menu first, else Back (a modal dialog's cancel; the key itself where the dialog could not be modal); a
    // dialog the browser closed by itself closes the view.
    this.listen(root, 'cancel', e => { e.preventDefault?.(); this.escape(); });
    this.listen(root, 'close', () => { if (this.isOpen && !this.root.open) this.close(); });   // not one queued before a re-open
    this.listen(root, 'keydown', e => { if (e.key !== 'Escape' || this.modal) return; e.preventDefault?.(); e.stopPropagation?.(); this.escape(); });
    this.listen(root, 'pointerdown', e => { if (this.menuOpen && !this.menu.contains?.(e.target) && !this.addButton.contains?.(e.target)) this.toggleMenu(false); });
    // Each canvas's size, as the browser lays it out (its plot fills what the column leaves).
    if (typeof globalThis.ResizeObserver === 'function') {
      this.observer = new globalThis.ResizeObserver(entries => {
        for (const entry of entries) {
          const col = [...this.cols.values()].find(c => c.part.plot === entry.target);
          const box = entry.contentRect;
          if (!col || !box) continue;
          const css = [Math.max(1, Math.round(box.width)), Math.max(1, Math.round(box.height))];
          if (col.css?.[0] === css[0] && col.css?.[1] === css[1]) continue;
          col.css = css; col.dirty = true;
        }
        this.wake();
      });
    }
  }
  /** One row of the key: its swatch and words (`tooltip`: its own, else the words); pointing at it (or focusing it)
   *  lights its cells in every column. */
  keyItem(by, value, color, text, tooltip = text) {
    const li = this.el('li', 'cmp-key-item'); li.dataset[by === 'kind' ? 'kind' : by === 'clusters' ? 'cluster' : 'step'] = String(value);
    const b = this.el('button', 'cmp-key-button'); b.type = 'button'; b.dataset.action = 'light';
    const sw = this.el('i', 'cmp-swatch'); sw.style.background = color;
    b.append(sw, this.el('span', '', text)); b.title = tooltip;
    b.onpointerenter = b.onfocus = () => this.setLit({ by, value });   // properties: a key rebuilt lets its handlers go
    b.onpointerleave = b.onblur = () => this.setLit(null);
    li.append(b);
    return li;
  }
  /** A column's DOM: its head (name, key and source, take out), its plot (the UMAP canvas, the wait line, how many cells
   *  it shows, the hover's tip) and its mixing (the number, the line at 100 transcripts each, the histogram, the stats). */
  column(key) {
    const el = (t, c, x) => this.el(t, c, x);
    const sec = el('section', 'cmp-col'); sec.dataset.compareColumn = key; sec.dataset.state = 'working';
    const head = el('header', 'cmp-col-head'), names = el('div', 'cmp-col-names');
    const title = el('h3', 'cmp-col-title'), sub = el('p', 'cmp-col-sub'), counted = el('p', 'cmp-col-count'); counted.hidden = true;
    names.append(title, sub, counted);
    const remove = this.button('', 'remove', 'cmp-tool', ICONS.close); remove.title = T.remove; remove.setAttribute('aria-label', T.remove);
    remove.onclick = event => { event?.stopPropagation?.(); this.remove(key); };
    head.append(names, remove);
    const note = el('p', 'cmp-col-note', T.smallCover); note.title = T.smallCoverTooltip; note.hidden = true;
    const plot = el('div', 'cmp-plot');
    const canvas = el('canvas', 'compare-umap'); canvas.setAttribute('role', 'img');
    const wait = el('p', 'cmp-wait', T.umapWaiting);
    const shown = el('p', 'cmp-shown'); shown.title = T.shownTooltip; shown.hidden = true;
    const tip = el('div', 'cmp-tip'); tip.hidden = true; tip.setAttribute('role', 'tooltip');
    const tipKind = el('p', 'cmp-tip-kind'), tipSwatch = el('i', 'cmp-swatch'), tipName = el('span', '');
    tipKind.append(tipSwatch, tipName);
    const tipShare = el('p', 'cmp-tip-share'), tipMore = el('p', 'cmp-tip-share cmp-tip-more'); tipMore.hidden = true;
    tip.append(tipKind, tipShare, tipMore);
    plot.append(canvas, wait, note, shown, tip);
    const mixing = el('div', 'cmp-mixing');
    const mixed = el('p', 'cmp-mixed'); mixed.dataset.mixed = '';
    const big = el('b', 'cmp-big'), tail = el('span', 'cmp-tail'), rough = el('span', 'cmp-rough', T.rough), mixedInfo = this.info();
    rough.title = T.roughTooltip; rough.hidden = true;
    mixed.append(big, ' ', tail, rough, mixedInfo);
    const like = el('p', 'cmp-like'); like.dataset.like100 = '';
    const likeText = el('span', ''), likeInfo = this.info();
    like.append(likeText, likeInfo);
    const hist = el('figure', 'cmp-hist'); hist.dataset.hist = ''; hist.title = T.histTooltip;
    const bars = el('div', 'cmp-bars'), ticks = el('div', 'cmp-ticks'), binNodes = [];
    BIN_LABELS.forEach((label, b) => {
      const bin = el('div', 'cmp-bin'); bin.dataset.bin = String(b);
      const fill = el('i', ''); fill.style.background = MIXING_COLORS[BIN_STEPS[b]]; fill.style.height = '0%';
      bin.append(fill); bars.append(bin); binNodes.push({ bin, fill });
      ticks.append(el('span', '', label));
    });
    hist.append(el('figcaption', 'cmp-hist-title', T.histTitle), bars, ticks);
    const stats = el('dl', 'cmp-stats'), stat = (label, tooltip) => {
      const d = el('div', 'cmp-stat'); d.title = tooltip;
      const value = el('dd', '', T.missing);
      d.append(value, el('dt', '', label)); stats.append(d);
      return value;
    };
    const cells = stat(T.statCells, T.statCellsTooltip), median = stat(T.statMedian, T.statMedianTooltip), under = stat(T.statUnder, T.statUnderTooltip);
    const error = el('p', 'cmp-col-error'); error.hidden = true;
    mixing.append(mixed, like, hist, stats, error);
    sec.append(head, plot, mixing);
    const part = { sec, title, sub, counted, remove, note, plot, canvas, wait, shown, tip, tipSwatch, tipName, tipShare, tipMore, mixed, big, tail, rough, mixedInfo,
      like, likeText, likeInfo, hist, bins: binNodes, stats, cells, median, under, error };
    canvas.onpointermove = e => this.pointer(key, e);   // properties: a column taken out lets its handlers go
    canvas.onpointerleave = () => this.setHover(null);
    return part;
  }

  // ---- opening and closing ------------------------------------------------------------------------------------------
  /** Opens the view on `run` (its id, or the run itself) at the frame on screen (`seq`, sent with every start: the run's
   *  column is that frame; `iteration` its iteration) of sample `sampleId`; `box` ([x0, y0, x1, y1] µm, optional) the
   *  run's region, for "covers under a tenth" until the engine has counted each column's squares. It lists the sample's
   *  segmentations, then starts the columns last chosen for the sample (else the run, Xenium's cells and the imported
   *  one). Returns whether it opened. */
  open({ run, iteration = null, sampleId = null, box = null, seq = null, kept = null } = {}) {
    const id = run && typeof run === 'object' ? run.id : run;
    if (id == null || id === '') return false;
    if (this.isOpen) this.close(false);
    this.isOpen = true; this.runId = String(id); this.sampleId = sampleId ?? null; this.iteration = finite(iteration) ? iteration : null;
    this.kept = Array.isArray(kept) ? kept.filter(k => typeof k === 'string') : [];
    this.seq = Number.isInteger(seq) && seq >= 0 ? seq : null;
    this.runBox = box4(box); this.clear();
    this.dpr = clamp(globalThis.devicePixelRatio || 1, 1, 3);
    this.showDialog();
    this.opened++; this.hook?.count('compareOpened');
    this.setState('starting');
    this.render();
    this.loadList(true);
    return true;
  }
  /** Shown as a modal dialog: the page underneath is inert and its keys quiet (main.js ignores keys while a dialog is
   *  open); where that cannot be, simply open. */
  showDialog() {
    const d = this.root;
    this.modal = false;
    try { if (!d.open) { if (typeof d.showModal === 'function') { d.showModal(); this.modal = true; } else { d.open = true; d.setAttribute('open', ''); } } }
    catch { d.open = true; d.setAttribute?.('open', ''); }
    try { d.focus?.({ preventScroll: true }); } catch { /* a convenience */ }
  }
  /** Esc: closes Add's menu when it is open (the focus back on Add), else the view (Back). */
  escape() {
    if (!this.isOpen) return;
    if (!this.menuOpen) { this.close(); return; }
    this.toggleMenu(false);
    try { this.addButton.focus?.({ preventScroll: true }); } catch { /* a convenience */ }
  }
  /** Closed (Back, Esc): nothing scheduled or asked any more, every column and picture let go; onClose tells the page. */
  close(tell = true) {
    if (!this.isOpen) return;
    this.isOpen = false; this.gen++; this.listGen++; this.setState('closed');
    for (const key of [...this.timers.keys()]) this.cancel(key);
    if (this.paintFrame != null) { this.clock.cancelAnimationFrame(this.paintFrame); this.paintFrame = null; }
    this.toggleMenu(false);
    this.clear(); this.syncColumns();
    try { if (this.root.open) { if (typeof this.root.close === 'function') this.root.close(); else { this.root.open = false; this.root.removeAttribute('open'); } } }
    catch { this.root.open = false; this.root.removeAttribute?.('open'); }
    if (tell) this.call('onClose');
  }
  /** What one opening holds: its columns, the status, the list, the layout and what the pointer does. */
  clear() {
    for (const col of this.cols.values()) this.dropColumn(col);
    this.cols = new Map(); this.keys = []; this.status = null; this.started = false; this.list = { state: 'idle', items: [] };
    this.layout = null; this.hover = null; this.lit = null; this.say(''); this.failures = 0; this.retryAt = -Infinity; this.fetchedAt = -Infinity;
    this.unreachable = false; this.failWhy = null; this.mismatches = 0; this.learning = null; this.menuOpen = false; this.kindsWhy = false;
    this.look = null; this.job = null; this.replaced = false; this.refusals = 0;
  }
  setState(state) {
    this.state = state;
    if (this.root && this.root.dataset.state !== state) this.root.dataset.state = state;
  }
  visibility() {
    if (!this.isOpen) return;
    if (this.hidden) {
      this.cancel('paint'); this.cancel('ask');
      if (this.paintFrame != null) { this.clock.cancelAnimationFrame(this.paintFrame); this.paintFrame = null; }
      return;
    }
    for (const col of this.cols.values()) col.dirty = true;
    this.wake(); this.ask(false); this.wantPoints();
  }
  /** The motion setting changed (calm motion, reduced motion): a fade on its way ends at once under either. */
  refresh() {
    if (!this.isOpen) return;
    if (this.still) for (const col of this.cols.values()) if (col.fade || col.flight) { col.fade = null; col.flight = null; col.dirty = true; }
    this.wake();
  }

  // ---- the segmentations and the columns chosen -----------------------------------------------------------------------
  /** Lists the sample's segmentations (for Add and the columns' names); the first time, then starts the columns. */
  async loadList(first = false) {
    const gen = ++this.listGen, sample = this.sampleId;
    this.list = { state: 'loading', items: this.list.items }; this.renderMenu();
    let items = null;
    try { items = typeof this.o.listSegmentations === 'function' ? normaliseList(await this.o.listSegmentations(sample)) : null; } catch { items = null; }
    if (gen !== this.listGen || !this.isOpen) return;
    this.list = items ? { state: 'ready', items } : { state: 'failed', items: [] };
    if (first) {
      const known = new Set(['run', ...this.list.items.map(s => s.key)]), kept = this.chosen.get(sample);
      const keys = kept ? kept.filter(k => this.list.state !== 'ready' || known.has(k)) : defaultColumns(this.list.items);
      this.begin(keys.length ? keys : ['run']);
    } else this.render();
    this.renderMenu();
  }
  /** A new compare of `columns` (keys, 'run' for the run's cells): the engine is asked to start (or reuse) it, every
   *  number shown so far goes (it was for another area), the pictures dim until the new layout comes. */
  async begin(columns) {
    if (!this.isOpen) return;
    const keys = [...new Set((columns ?? []).filter(k => typeof k === 'string' && k))].slice(0, MAX_COLUMNS);
    if (!keys.length) keys.push('run');
    const gen = ++this.gen;
    this.keys = keys; if (this.sampleId != null) this.chosen.set(this.sampleId, [...keys]);
    this.status = null; this.started = false; this.failures = 0; this.retryAt = -Infinity; this.fetchedAt = -Infinity;
    this.unreachable = false; this.failWhy = null; this.mismatches = 0; this.layout = null; this.say('');
    this.job = null; this.replaced = false; this.refusals = 0;
    this.cancel('ask');
    for (const [key, col] of this.cols) {
      if (!keys.includes(key)) continue;
      this.cancel(`points:${key}`);
      if (col.comp) { col.ghost = col.comp.canvas; col.comp = null; }
      // A download on its way for the compare before is let go (its token no longer the column's: download()).
      Object.assign(col, { data: null, points: null, own: null, clusters: null, slots: null, ownFailed: false, want: null, tries: 0, failed: false, loading: null,
        layoutId: null, grid: null, fade: null, flight: null, dirty: true });
    }
    if (this.hover) this.setHover(null);
    this.syncColumns();
    this.setState('starting'); this.render(); this.wake();
    let reply;
    try { this.counted(); reply = await this.o.start(this.runId, [...keys], this.seq); } catch (error) {
      if (gen === this.gen && this.isOpen) this.fail(plain(error?.message));
      return;
    }
    if (gen !== this.gen || !this.isOpen) return;
    // The columns the engine took (the reply's, when it names them: keys, or columns with a key), in its order.
    const took = Array.isArray(reply?.columns) ? reply.columns.map(c => (typeof c === 'string' ? c : c?.key)).filter(k => typeof k === 'string' && k) : [];
    if (took.length && took.length <= MAX_COLUMNS && (took.length !== keys.length || took.some((k, i) => k !== keys[i]))) {
      this.keys = [...new Set(took)]; if (this.sampleId != null) this.chosen.set(this.sampleId, [...this.keys]);
      this.syncColumns();
    }
    this.started = true;
    this.setState('working');
    // The reply is the compare as it stands (engine/compare.py: a reused one may be ready at once): taken as a status.
    // Its job is this view's comparison from now on: a status of another job is another window's (take()).
    const first = normaliseStatus(reply);
    this.job = first?.job ?? null;
    if (first?.columns.length) this.take(first); else this.render();
    this.ask(true);
  }
  /** The compare could not start or failed: said with the engine's words when it gave any, and Try again. */
  fail(why) {
    this.failWhy = why || ''; this.setState('failed'); this.cancel('ask');
    this.render();
  }
  /** A line in the alert under the header (a press that explains), '' to clear it. */
  say(text) { this.note = text || ''; if (this.alertText) this.renderAlert(); }
  /** Add a column (the menu): the fifth explains. */
  add(key) {
    if (!this.isOpen || this.keys.includes(key)) return;
    if (this.keys.length >= MAX_COLUMNS) { this.say(T.addFull); return; }
    this.toggleMenu(false);
    this.begin([...this.keys, key]);
  }
  /** Take a column out: the last one explains. */
  remove(key) {
    if (!this.isOpen || !this.keys.includes(key)) return;
    if (this.keys.length <= 1) { this.say(T.keepOne); return; }
    this.begin(this.keys.filter(k => k !== key));
  }
  /** The columns' DOM as the keys stand: new ones made, gone ones let go, in the keys' order. */
  syncColumns() {
    for (const [key, col] of [...this.cols]) if (!this.keys.includes(key)) { this.dropColumn(col); this.cols.delete(key); }
    for (const key of this.keys) {
      if (this.cols.has(key)) continue;
      const part = this.column(key);
      this.cols.set(key, { key, part, data: null, points: null, own: null, clusters: null, slots: null, ownFailed: false, want: null, tries: 0, failed: false,
        loading: null, layoutId: null, comp: null, ghost: null, fade: null, flight: null, fadeCanvas: null, grid: null, css: null, dirty: true });
      this.observer?.observe?.(part.plot);
    }
    const order = this.keys.map(k => this.cols.get(k).part.sec);
    if (order.length !== this.grid.children.length || order.some((n, i) => this.grid.children[i] !== n)) this.grid.replaceChildren(...order);
    if (this.grid.dataset.count !== String(order.length)) this.grid.dataset.count = String(order.length);
  }
  dropColumn(col) {
    this.cancel(`points:${col.key}`);
    this.observer?.unobserve?.(col.part.plot);
    col.part.sec.remove?.();
    col.comp = null; col.ghost = null; col.fade = null; col.flight = null; col.fadeCanvas = null; col.points = null; col.own = null; col.clusters = null;
    col.slots = null; col.grid = null; col.img = null;
  }

  // ---- asking for the status ------------------------------------------------------------------------------------------
  wantsStatus() {
    return this.isOpen && !this.hidden && this.started && !this.unreachable && this.state === 'working' && typeof this.o.fetchStatus === 'function';
  }
  /** Ask for the status when it is wanted: never sooner than STATUS_MS after the last ask (nor before a retry is due),
   *  one at a time. `now`: the timer fired (else only scheduled). */
  ask(now) {
    if (!this.wantsStatus() || this.asking) return;
    const wait = Math.max(this.fetchedAt + STATUS_MS, this.retryAt) - this.now();
    if (!now || wait > 0) { if (!this.timers.has('ask')) this.later('ask', wait, () => this.ask(true)); return; }
    this.askNow();
  }
  async askNow() {
    const gen = this.gen, run = this.runId;
    this.asking = true; this.fetchedAt = this.now(); this.polls++; this.counted();
    let status = null;
    try { status = normaliseStatus(await this.o.fetchStatus(run)); } catch { status = null; }   // quiet: api.js's /compare rule
    this.asking = false;
    if (gen === this.gen && this.isOpen && this.state === 'working') {
      if (status) { this.failures = 0; this.retryAt = -Infinity; this.take(status); }
      else {
        this.failures++;
        if (this.failures >= FAIL_TRIES) { this.unreachable = true; this.render(); }
        else { this.retryAt = this.now() + RETRY_MS * 2 ** (this.failures - 1); setText(this.progressNode, this.progressText()); }
      }
    }
    this.ask(false);
  }
  /** A status: kept when it is of the columns asked for (the engine may still answer for the compare before: asked again,
   *  MISMATCH_TRIES times at the most), its numbers into their columns, the points of a new layout asked for. */
  take(status) {
    if (this.job && status.job && status.job !== this.job) { this.replacedElsewhere(); return; }
    const keys = status.columns.map(c => c.key);
    if (keys.length && (keys.length !== this.keys.length || keys.some(k => !this.keys.includes(k)))) {
      if (++this.mismatches >= MISMATCH_TRIES) this.fail('');
      return;
    }
    this.mismatches = 0; this.status = status;
    for (const c of status.columns) { const col = this.cols.get(c.key); if (col) col.data = c; }
    if (status.kinds) this.kindsWhy = false;
    if (status.state === 'failed') { this.failWhy = status.error ?? ''; this.cancel('ask'); }
    this.setState(status.state);
    this.relook();
    this.render();
    this.wantPoints();
  }
  /** Another window replaced this view's comparison (the engine keeps one per run): nothing more is asked or drawn for
   *  it; the view says so, with Try again (this view's columns compared anew). */
  replacedElsewhere() {
    this.replaced = true; this.setState('failed'); this.cancel('ask');
    for (const col of this.cols.values()) { this.cancel(`points:${col.key}`); col.loading = null; }
    this.render();
  }
  /** The kinds came (or their colours changed) after a layout was drawn: its pictures cross-fade to the new colours. */
  relook() {
    const look = this.lookNow();
    if (this.look != null && look !== this.look) for (const col of this.cols.values()) if (col.layoutId && col.layoutId === this.layout?.id) this.startFade(col);
    this.look = look;
    this.wake();
  }

  // ---- the UMAP points ------------------------------------------------------------------------------------------------
  /** The colours on screen: the colouring and the kinds' colours. */
  lookNow() { return JSON.stringify([this.colouring(), this.status?.kinds?.map(k => [k.index, k.color]) ?? null]); }
  /** The layout the status describes: the model, how many layouts so far, the area, the columns and their points. */
  layoutKey() {
    const s = this.status;
    return s ? JSON.stringify([s.model, s.layout, s.area?.box ?? null, s.columns.map(c => [c.key, c.points]), s.job]) : null;
  }
  /** Each column with points in the status gets them, once per layout (POINT_TRIES asks at the most), one download at a
   *  time per column (`loading`: the one on its way). */
  wantPoints() {
    if (!this.isOpen || this.hidden || !this.status || typeof this.o.fetchPoints !== 'function') return;
    const want = this.layoutKey();
    this.status.columns.forEach((c, index) => {
      const col = this.cols.get(c.key);
      if (!col || c.points <= 0) return;
      if (col.want !== want) Object.assign(col, { want, tries: 0, failed: false });
      if (col.points && col.pointsKey === want) return;
      if (col.loading || col.failed || this.timers.has(`points:${col.key}`)) return;
      this.download(col, index, want, this.status.job, this.status.layout);
    });
    this.maybeLayout();
  }
  /** Column `index`'s points for layout `want` (the comparison `job` at its `layout`: the engine refuses those of another
   *  with 409). The download on its way is the column's `loading` (a token of its own): one that a new compare let go
   *  (begin) changes nothing when it lands; one whose layout moved on meanwhile (the engine's next layout came with the
   *  status that said ready: no status will ask again) has the new one asked for. A refusal keeps no bytes: the status
   *  is asked for again (another window's comparison, or a newer layout). */
  async download(col, index, want, job = null, layout = null) {
    const gen = this.gen, token = {}, own = !!this.status?.columns?.[index]?.own;
    col.loading = token; col.tries++;
    // With the points, the column's own map and its clusters when the engine made them (the same points, same order).
    const get = what => {
      this.downloads++; this.counted();
      return Promise.resolve().then(() => this.o.fetchPoints(this.runId, index, job, layout, ...(what ? [what] : [])));
    };
    const [a, b, c] = await Promise.allSettled([get(null), own ? get('own') : null, own ? get('clusters') : null]);
    if (col.loading !== token) return;
    col.loading = null;
    if (gen !== this.gen || !this.isOpen || this.cols.get(col.key) !== col) return;
    const refused = [a, b, c].some(r => r.status === 'rejected' && r.reason?.status === 409);
    if (refused) { col.tries--; this.recheck(want); return; }
    const points = a.status === 'fulfilled' ? decodePoints(a.value) : null;
    if (points) this.refusals = 0;
    if (col.want !== want) { this.wantPoints(); return; }
    if (points) {
      let mine = own && b.status === 'fulfilled' ? decodePoints(b.value) : null;
      const ids = mine && mine.n === points.n && c.status === 'fulfilled' ? decodeClusters(c.value, points.n) : null;
      if (!ids) mine = null;
      Object.assign(col, { points, pointsKey: want, failed: false, own: mine, clusters: ids, slots: null, ownFailed: own && !mine });
    }
    else if (col.tries >= POINT_TRIES) col.failed = true;
    else this.later(`points:${col.key}`, RETRY_MS * col.tries, () => this.wantPoints());
    this.maybeLayout();
    this.render();
  }
  /** Points refused for the comparison and layout `want` named: the status is asked for again at once. One ask covers
   *  every column refused meanwhile, and only the asks count (MISMATCH_TRIES in a row at the most, then the view says it
   *  could not compare), so four columns refused in one race count once; a newer status already here has the column's
   *  points asked for with it. */
  recheck(want) {
    if (this.status && want !== this.layoutKey()) { this.wantPoints(); return; }
    if (this.state !== 'working') this.setState('working');
    if (this.asking) return;              // its answer asks for the points again (take, wantPoints)
    if (++this.refusals >= MISMATCH_TRIES) { this.fail(''); return; }
    this.cancel('ask'); this.fetchedAt = -Infinity; this.retryAt = -Infinity;
    this.ask(true);
  }
  /** Once every column with points has them (or could not get them), one scale for all of them: the layout arrives and
   *  each column's picture cross-fades in. */
  maybeLayout() {
    const s = this.status;
    if (!s) return;
    const want = this.layoutKey(), withPoints = s.columns.filter(c => c.points > 0).map(c => this.cols.get(c.key)).filter(Boolean);
    if (!withPoints.length || withPoints.some(col => !(col.points && col.pointsKey === want) && !col.failed)) return;
    const ready = withPoints.filter(col => col.points && col.pointsKey === want);
    if (!ready.length || this.layout?.id === want) return;
    // One scale for every column: the engine's view box of the layout (its 2nd-98th percentiles) widened by FIT_MARGIN,
    // else read from the points themselves.
    const bounds = s.bounds ? widen(s.bounds) : fitLayout(ready.map(col => col.points));
    if (!bounds) return;
    const most = Math.max(...ready.map(col => col.points.n));
    // Each its own: one scale for every column's own map too (the engine's view box of them all, else their points').
    const own = ready.map(col => col.own).filter(Boolean), ownBounds = !own.length ? null : s.ownBounds ? widen(s.ownBounds) : fitLayout(own);
    this.layout = { id: want, bounds, ownBounds, most, diff: null, clusterIds: null };
    if (this.hover) this.setHover(null);
    for (const col of ready) { col.layoutId = want; col.grid = null; col.flight = null; this.startFade(col); col.ghost = null; }
    this.look = this.lookNow();
    this.render(); this.wake();
  }

  // ---- presses ----------------------------------------------------------------------------------------------------------
  /** How the dots are coloured now: by kind only when the model has kinds; the third choice Clusters on each its own,
   *  Differences on one map. */
  colouring() {
    const by = this.colourBy ?? 'kind';
    if (by === 'groups') return this.mapMode === 'own' ? 'clusters' : 'differences';
    return this.status?.kinds && by === 'kind' ? 'kind' : 'mixing';
  }
  /** The Map switch: each its own or one map, remembered in this browser. The dots on screen fly from the map they are
   *  on to the other (the same points in the same order); a column with no map of the other kind cross-fades. */
  setMap(mode) {
    if (!this.isOpen || !MAPS.includes(mode) || mode === this.mapMode) return;
    const before = new Map([...this.cols.values()].map(col => [col.key, this.shown(col)]));
    this.mapMode = mode; this.saveMap();
    if (this.hover) this.setHover(null);
    this.lit = null;
    this.look = this.lookNow();
    this.flyTo(before);
    this.render(); this.wake();
  }
  /** After a press of the Map switch: each column drawn before and now on the two maps flies between them, in the colours
   *  it lands in; any other cross-fades (reduced or calm motion: one paint). */
  flyTo(before) {
    const now = this.now(), by = this.colouring(), pal = palettes(this.status?.kinds ?? null);
    for (const col of this.cols.values()) {
      const from = before.get(col.key) ?? null, to = this.shown(col);
      col.flight = null; col.dirty = true;
      if (this.still || (!from && !to)) { col.fade = null; continue; }
      if (from && to && from.own !== to.own && from.pts.n === to.pts.n) {
        col.fade = null; col.flight = { from, to, start: now, ...this.slotsOf(col, to, by, pal), xy: null };
      } else this.startFade(col);
    }
  }
  /** What a column draws on the layout on screen: {pts, bounds, own}: its own map under each its own once it has one,
   *  else the one map; null for none (each its own on a column with no map of its own). A layout with no own maps at all
   *  is drawn on the one map while they may still come: the engine works, or a newer layout (the last call's, which
   *  brings them) is on its way. */
  shown(col) {
    const lay = this.layout;
    if (!col?.points || !lay || col.layoutId !== lay.id) return null;
    if (this.mapMode === 'own') {
      if (col.own && lay.ownBounds) return { pts: col.own, bounds: lay.ownBounds, own: true };
      if (lay.ownBounds || (this.state !== 'working' && lay.id === this.layoutKey())) return null;
    }
    return { pts: col.points, bounds: lay.bounds, own: false };
  }
  /** The yardstick column (each own map is lined up with it, Differences set the others beside it): the run's when it
   *  is a column, else the first. */
  yardstick() { return this.keys.includes('run') ? 'run' : this.keys[0] ?? null; }
  /** The yardstick in words: as a column ("the run's column") or as an owner ("the run's", diffTip). */
  refName(owner = false) { return this.yardstick() === 'run' ? (owner ? T.refRun : T.refRunColumn) : owner ? T.refFirst : T.refFirstColumn; }
  /** A column's Differences on the layout on screen (worked out for every column at once, the first time asked). */
  diffOf(col) {
    const lay = this.layout;
    if (!lay) return null;
    if (!lay.diff) {
      const sets = this.keys.map(key => { const c = this.cols.get(key); return c?.points && c.layoutId === lay.id ? c.points : null; });
      const out = differences(sets, Math.max(0, this.keys.indexOf(this.yardstick())), lay.bounds);
      lay.diff = new Map(this.keys.map((key, i) => [key, out[i]]));
    }
    return lay.diff.get(col.key) ?? null;
  }
  /** Each point's colour slot and the palette for colouring `by` (Clusters: the column's own clusters on its own map;
   *  Differences: its steps on the one map; grey where there are none). */
  slotsOf(col, view, by, pal) {
    const pts = view.pts;
    if (by === 'kind') return { slots: pts.kind, palette: pal.kind };
    if (by === 'clusters') return { slots: view.own && col.clusters ? (col.slots ??= clusterSlots(col.clusters)) : new Uint8Array(pts.n).fill(255), palette: pal.clusters };
    if (by === 'differences') {
      const d = view.own ? null : this.diffOf(col);
      return { slots: d ? d.steps : new Uint8Array(pts.n).fill(DIFF_SAME), palette: pal.diff };
    }
    return { slots: Uint8Array.from(pts.share, stepOf), palette: pal.mixing };
  }
  /** Colour by Kind or Mixing (never greyed out: Kind without kinds says why and stays by mixing), or the third choice. */
  setColour(by) {
    if (!this.isOpen) return;
    // Kind without the model's kinds: said in the key (and taken as soon as they come); the dots stay by mixing.
    this.colourBy = by; this.kindsWhy = by === 'kind' && !this.status?.kinds;
    if (this.lit) { this.lit = null; for (const col of this.cols.values()) if (col.layoutId) col.dirty = true; }
    this.relook();
    this.renderColour(); this.renderKey(); this.wake();
  }
  /** A kind (or a step of mixing) lit from the key: its cells over the others, which dim; null for none. */
  setLit(lit) {
    const key = lit ? `${lit.by}:${lit.value}` : '';
    if (key === (this.lit ? `${this.lit.by}:${this.lit.value}` : '')) return;
    this.lit = lit && lit.by === this.colouring() ? lit : null;
    for (const col of this.cols.values()) if (col.layoutId) col.dirty = true;
    this.wake();
  }
  toggleMenu(open = !this.menuOpen) {
    if (open === this.menuOpen) return;
    this.menuOpen = open; this.menu.hidden = !open; this.addButton.setAttribute('aria-expanded', String(open));
    if (open) { this.say(''); this.renderMenu(); }
  }
  /** Learn the details (the model's kinds need them): asked once; the view says the rest. */
  async learn() {
    const sample = this.sampleId;
    if (!this.isOpen || typeof this.o.learnDetails !== 'function' || this.learning === 'asking') return;
    this.learning = 'asking'; this.renderKey();
    let answer = null, ok = true;
    try { answer = await this.o.learnDetails(sample); } catch { ok = false; }
    if (!this.isOpen || sample !== this.sampleId) return;
    const ready = ok && (answer?.model?.deep === true || answer?.detail?.state === 'ready');
    this.learning = ok ? (ready ? null : 'learning') : 'failed';
    if (ready) this.begin(this.keys); else this.renderKey();
  }

  // ---- the pointer --------------------------------------------------------------------------------------------------------
  /** The dot under the pointer (HOVER_PX), named in the tip; its place ringed in every column (the same place is the
   *  same kind of cell in all of them). */
  pointer(key, e) {
    const col = this.cols.get(key), comp = col?.comp, place = comp?.place;
    if (!place || !comp.pts || col.flight || !finite(e?.offsetX) || !finite(e?.offsetY)) { this.setHover(null); return; }
    const dpr = this.dpr, x = e.offsetX * dpr, y = e.offsetY * dpr;
    col.grid ??= pointGrid(comp.pts, place, GRID_PX * dpr);
    const index = nearestPoint(comp.pts, place, col.grid, x, y, HOVER_PX * dpr);
    this.setHover(index == null ? null : { key, index, x: e.offsetX, y: e.offsetY });
  }
  setHover(hover) {
    const was = this.hover;
    if (!hover && !was) return;
    if (hover && was && hover.key === was.key && hover.index === was.index) { this.hover = hover; this.placeTip(); return; }
    this.hover = hover;
    this.renderTip();
    for (const col of this.cols.values()) if (col.layoutId) col.dirty = true;
    this.wake();
  }

  // ---- words --------------------------------------------------------------------------------------------------------------
  render() {
    if (!this.isOpen) return;
    const s = this.status;
    // The area: what all of them cover (one column: what it covers), its squares in the ⓘ.
    const size = s?.area?.size_um;
    const area = size ? compare(this.keys.length > 1 ? 'areaAll' : 'areaOne', { size: sizeText(size) })
      : this.state === 'failed' || this.unreachable ? '' : T.areaFinding;
    setText(this.areaText, area); setHidden(this.areaNode, !area);
    const tip = size && s.area.tiles != null ? compare('areaTooltip', { n: number(s.area.tiles) }) : '';
    setTitle(this.areaInfo, tip); this.areaInfo.setAttribute('aria-label', tip); setHidden(this.areaInfo, !tip);
    setData(this.areaNode, 'compareArea', size ? size.map(v => String(v)).join(',') : '');
    setText(this.progressNode, this.progressText());
    this.renderAlert();
    const covers = this.covers();
    for (const key of this.keys) this.renderColumn(this.cols.get(key), covers);
    this.renderHistograms();
    this.renderColour(); this.renderKey();
    if (this.menuOpen) this.renderMenu();
  }
  /** "Reading each column's counts… 2 of 5" while the engine works; nothing once it is done. */
  progressText() {
    if (this.unreachable || this.state === 'failed' || this.state === 'ready' || this.state === 'closed') return '';
    if (this.failures > 0) return T.retrying;
    if (this.state === 'starting' || !this.status) return T.starting;
    const s = this.status, word = STEPS[s.step] ? T[STEPS[s.step]] : null;
    if (!word) return T.working;
    return s.progress ? compare('progress', { step: word, done: number(s.progress.done), of: number(s.progress.of) }) : word;
  }
  renderAlert() {
    let text = this.note, retry = false;
    if (this.replaced) { text = REPLACED_TEXT; retry = true; }
    else if (this.unreachable) { text = T.unreachable; retry = true; }
    else if (this.state === 'failed') { text = this.failWhy ? compare('failedWhy', { why: this.failWhy }) : T.failed; retry = true; }
    setText(this.alertText, text); setTitle(this.alertText, text); setHidden(this.retryButton, !retry); setHidden(this.alert, !text);
  }
  /** Which columns cover under a tenth of another's area: by the 100 µm squares each covers by itself (the engine's
   *  `tiles`, once it has found the area), else by the extents of the sample's segmentations and the run's region (their
   *  boxes, before the area is known). With the run a column, the engine reads every column inside the run's region
   *  only: the extents are cut to it too. */
  covers() {
    const tiles = this.keys.map(key => this.cols.get(key)?.data?.tiles ?? null), limit = this.keys.includes('run') ? this.runBox : null;
    const areas = tiles.some(t => t != null) ? tiles
      : this.keys.map(key => (key === 'run' ? area4(this.runBox) : area4(within(this.item(key)?.extent, limit))));
    return new Set(this.keys.filter((key, i) => areas[i] != null && areas.some((a, j) => j !== i && a != null && areas[i] < SMALL_COVER * a)));
  }
  /** The counting rule in words (the number's ⓘ): each column on its own count, and for a segmentation with none of its
   *  own as the engine counted it (its `rule`, the run's own input filters: quality and the genes the run left out),
   *  quality MIN_QV without one; then, when a column is on proseg's count, that count's own bias (countingBias). */
  /** A count's label's tooltip: what the count is (its tooltip's key: countKeys), then the bias of proseg's count while a
   *  column is on it (as countingText ends). */
  countTip(tip) {
    const bias = this.status?.columns?.some(c => c.count === 'proseg') ? T.countingBias : '';
    return [T[tip], bias].filter(Boolean).join(' ');
  }
  countingText() {
    const rule = this.status?.rule, q = rule?.minQv ?? MIN_QV;
    const text = compare(rule?.exclude ? 'countingExcluded' : 'counting', { q: number(q, Number.isInteger(q) ? 0 : 1) });
    return this.status?.columns?.some(c => c.count === 'proseg') ? `${text} ${T.countingBias}` : text;
  }
  item(key) { return this.list.items.find(s => s.key === key) ?? null; }
  /** A column's name (the engine's, else the sample's list's, else its key; the run's: "This run · iteration 450") and its
   *  second line (its key and what made it). */
  names(key, data) {
    if (key === 'run') {
      const name = data?.name || (this.iteration != null ? compare('runName', { i: number(this.iteration) }) : T.runNameNoIter);
      return { name, sub: T.sourceProseg };
    }
    const s = this.item(key), name = data?.name || s?.name || key;
    const word = SOURCES[s?.source] ? T[SOURCES[s.source]] : '';
    const source = word && s?.model ? compare('withModel', { source: word, model: s.model }) : word;
    return { name, sub: name === key ? source : source ? compare('colSub', { key, source }) : key };
  }
  renderColumn(col, covers) {
    if (!col) return;
    const p = col.part, d = col.data, { name, sub } = this.names(col.key, d);
    setText(p.title, name); setTitle(p.title, name); setText(p.sub, sub); setTitle(p.sub, sub); setHidden(p.sub, !sub);
    // Which count it is read on, under its name; what that count is in its tooltip, with the bias of proseg's count
    // while a column is on it (the label can be cut short in a narrow column).
    const keys = countKeys(col.key, d?.count, d?.form), counted = keys ? T[keys[0]] : '';
    setText(p.counted, counted); setHidden(p.counted, !counted); setTitle(p.counted, counted ? this.countTip(keys[1]) : '');
    const label = compare('umapLabel', { name });
    if (p.canvas.getAttribute('aria-label') !== label) p.canvas.setAttribute('aria-label', label);
    setHidden(p.note, !covers.has(col.key));
    // A column the engine could not work out says why (its words); a compare that failed as a whole, or an engine that
    // isn't answering, says it once above the columns (the alert), the columns without numbers then quiet.
    const failed = d?.state === 'failed', stopped = this.state === 'failed' || this.unreachable;
    setData(p.sec, 'state', failed ? 'failed' : d?.state ?? (this.state === 'starting' ? 'starting' : stopped ? 'stopped' : 'working'));
    // The number: "13 in 100 cells look mixed" (whole, as the mixing check rounds it), "rough" under 300 cells judged;
    // its ⓘ: the cells it is of, how every column is counted, the model's caveat, why no verdict.
    const known = d?.mixed != null && d?.judged != null;
    if (known && d.judged >= HEADLINE_MIN_JUDGED) {
      setText(p.big, whole(d.mixed)); setText(p.tail, T.mixedTail);
    } else if (d?.judged != null && d.judged < HEADLINE_MIN_JUDGED) {
      setText(p.big, ''); setText(p.tail, compare('mixedTooFew', { j: number(d.judged) }));
    } else {
      setText(p.big, '');
      setText(p.tail, failed || stopped ? '' : T[COLUMN_STATES[d?.state]] ?? T.columnWorking);
    }
    setHidden(p.big, !p.big.textContent);
    setHidden(p.rough, !(known && d.judged >= HEADLINE_MIN_JUDGED && d.judged < ROUGH_JUDGED));
    const mixedTip = d?.judged != null ? [compare('mixedTooltip', { j: number(d.judged) }), this.countingText(), T.transfer, T.sideBySide].join('\n') : '';
    setTitle(p.mixedInfo, mixedTip); p.mixedInfo.setAttribute('aria-label', mixedTip); setHidden(p.mixedInfo, !mixedTip);
    setData(p.mixed, 'mixed', d?.mixed != null ? String(d.mixed) : '');
    setHidden(p.mixed, !p.tail.textContent);
    // Like for like: "At 100 transcripts each: 11 in 100"; its tooltip says whose draw the run column's cut starts from
    // (H6) and, on a kind of run whose cut depth is kept from before proseg's count (like_depth), that too.
    const like = d?.like100 != null && d?.judged != null && d.judged >= HEADLINE_MIN_JUDGED;
    setText(p.likeText, like ? compare('like', { n: whole(d.like100) }) : '');
    const likeTip = like ? [compare('likeTooltip', { j: number(d.likeCells ?? d.judged) }), d.likeFrom ? T[LIKE_FROM[d.likeFrom]] : '',
      this.kept.includes('like_depth') ? QC_TEXT.notRemeasuredShort : ''].filter(Boolean).join(' ') : '';
    setTitle(p.likeInfo, likeTip); p.likeInfo.setAttribute('aria-label', likeTip);
    setData(p.like, 'like100', d?.like100 != null ? String(d.like100) : '');
    setHidden(p.like, !like);
    // The histogram (its bars drawn with every column's: renderHistograms) and the stats.
    setData(p.hist, 'hist', d?.histogram ? d.histogram.join(',') : '');
    setData(p.sec, 'judged', d?.judged != null ? String(d.judged) : '');
    setHidden(p.hist, !d?.histogram);
    setText(p.cells, d?.cells != null ? number(d.cells) : T.missing);
    setText(p.median, d?.medianTx != null ? number(Math.round(d.medianTx)) : T.missing);
    setText(p.under, d?.under100 != null ? compare('percent', { p: number(Math.round(d.under100 * 100)) }) : T.missing);
    setHidden(p.stats, failed || (d?.cells == null && d?.medianTx == null && d?.under100 == null));
    const error = failed ? d?.error || T.columnFailed : '';
    setText(p.error, error); setHidden(p.error, !error);
    // The plot: its wait line until its picture (too few cells for one only once the whole compare is done: the layout
    // comes after every column's numbers), how many cells it shows when the engine picked some.
    const view = this.shown(col), drawn = !!view;
    const layoutError = this.status?.layoutError, none = !layoutError || layoutError === 'too_few_cells' ? T.umapNone : T.layoutFailed;
    // Each its own on a column with points but no map of its own: too few cells for one, or it could not be laid out.
    const ownError = this.status?.ownError, noOwn = !!col.points && col.layoutId === this.layout?.id && this.mapMode === 'own'
      ? (col.ownFailed || (ownError && ownError !== 'too_few_cells') ? T.ownFailed : T.ownNone) : '';
    const wait = drawn || failed || (stopped && !col.points) ? '' : noOwn || (col.failed ? T.umapFailed
      : this.state === 'ready' && d && d.points <= 0 ? none : T.umapWaiting);
    setText(p.wait, wait); setHidden(p.wait, !wait);
    const shown = drawn && d?.sampled && d.judged != null ? compare('shown', { n: number(view.pts.n), m: number(d.judged) }) : '';
    setText(p.shown, shown); setHidden(p.shown, !shown);
    setData(p.sec, 'points', drawn ? String(view.pts.n) : '0');
  }
  /** Every column's histogram on one scale (the tallest bin of any column fills its height), each bin's tooltip its
   *  share of that column's judged cells. */
  renderHistograms() {
    const shares = new Map(this.keys.map(key => [key, binShares(this.cols.get(key)?.data?.histogram)]));
    const top = Math.max(0, ...[...shares.values()].flatMap(s => s ?? []));
    for (const key of this.keys) {
      const col = this.cols.get(key), s = shares.get(key), hist = col?.data?.histogram;
      if (!col) continue;
      col.part.bins.forEach(({ bin, fill }, b) => {
        const height = s && top > 0 ? `${((s[b] / top) * 100).toFixed(1)}%` : '0%';
        if (fill.style.height !== height) fill.style.height = height;
        setTitle(bin, s ? compare('bar', { bin: BIN_LABELS[b], p: number(Math.round(s[b] * 100)), n: number(hist[b]) }) : '');
        setData(bin, 'count', hist ? String(hist[b]) : '');
      });
    }
  }
  /** The pressed button is the colouring on screen (Kind pressed without kinds leaves Mixing pressed: the key says why). */
  renderColour() {
    const by = this.colouring(), own = this.mapMode === 'own', group = own ? 'clusters' : 'differences', ref = { ref: this.refName() };
    setText(this.groupLabel, own ? T.byClusters : T.byDifferences);
    setTitle(this.groupButton, compare(own ? 'byClustersTooltip' : 'byDifferencesTooltip', ref));
    setData(this.groupButton, 'colourBy', group);
    setTitle(this.ownButton, compare('mapOwnTooltip', ref));
    for (const [b, want] of [[this.kindButton, by === 'kind'], [this.mixingButton, by === 'mixing'], [this.groupButton, by === group],
      [this.oneButton, !own], [this.ownButton, own]]) {
      const v = String(want);
      if (b.getAttribute('aria-pressed') !== v) b.setAttribute('aria-pressed', v);
    }
    setData(this.root, 'colourBy', this.isOpen ? by : '');
    setData(this.root, 'map', this.isOpen ? this.mapMode : '');
  }
  /** The key: the kinds (by kind) or the three steps of mixing; without kinds once the engine is done (or Kind pressed),
   *  why, with Learn the details. */
  renderKey() {
    const s = this.status, by = this.colouring(), kinds = s?.kinds;
    setHidden(this.kindsKey, by !== 'kind'); setHidden(this.mixKey, by !== 'mixing');
    setHidden(this.clustersKey, by !== 'clusters'); setHidden(this.diffKey, by !== 'differences');
    const ref = { ref: this.refName() }, clustersTip = compare('clustersKeyTooltip', ref), diffTip = compare('diffKeyTooltip', ref);
    setTitle(this.clustersInfo, clustersTip); this.clustersInfo.setAttribute('aria-label', clustersTip);
    setTitle(this.diffInfo, diffTip); this.diffInfo.setAttribute('aria-label', diffTip);
    if (by === 'clusters') this.renderClusters();
    const kKey = kinds ? kinds.map(k => `${k.index}\u0000${k.name}\u0000${k.color}`).join('\u0001') : '';
    if (kKey !== this.kindsKeyText) {
      this.kindsKeyText = kKey;
      this.kindsList.replaceChildren(...(kinds ?? []).slice().sort((a, b) => a.index - b.index).map(k => this.keyItem('kind', k.index, k.color, k.name || T.noKind)));
    }
    const done = this.state === 'ready' || this.state === 'failed';
    // Why no kinds (the engine's kinds_reason): the model has no details yet (needs_detail, or an engine that does not
    // say): Learn the details; the sample was being saved (busy): a new Compare makes them; it could not make them
    // (no_source, failed): said, nothing to press.
    const needsDetail = !s?.kindsReason || s.kindsReason === 'needs_detail';
    let note = '';
    if (s && !kinds && (done || this.kindsWhy)) {
      note = this.learning === 'learning' ? T.learning : this.learning === 'failed' ? T.learnFailed : !done ? T.kindsComing
        : needsDetail ? T.kindsNeeded : s.kindsReason === 'busy' ? T.kindsBusy : T.kindsUnavailable;
    } else if (!s && this.kindsWhy) note = T.kindsComing;
    const learn = !!note && done && !kinds && needsDetail && typeof this.o.learnDetails === 'function' && this.learning !== 'learning';
    setText(this.keyNoteText, note); setHidden(this.keyNote, !note); setHidden(this.learnButton, !learn);
    if (learn && (this.learning === 'asking') !== (this.learnButton.getAttribute('aria-busy') === 'true')) {
      if (this.learning === 'asking') this.learnButton.setAttribute('aria-busy', 'true'); else this.learnButton.removeAttribute('aria-busy');
    }
  }
  /** The clusters' key: each number on screen (each column's own clusters, from 1), in its colour, its tooltip naming the
   *  columns it is in; pointing at one lights it in every column. Worked out once per layout. */
  renderClusters() {
    const lay = this.layout;
    if (lay && !lay.clusterIds) {
      const where = new Map();
      for (const key of this.keys) {
        const col = this.cols.get(key);
        if (!col?.own || !col.clusters || col.layoutId !== lay.id) continue;
        const name = this.names(key, col.data).name;
        for (const id of new Set(col.clusters)) if (id < 255) { if (!where.has(id)) where.set(id, []); where.get(id).push(name); }
      }
      lay.clusterIds = [...where.keys()].sort((a, b) => a - b).map(id => [id, where.get(id)]);
    }
    const ids = lay?.clusterIds ?? [], text = JSON.stringify(ids);
    if (text === this.clustersKeyText) return;
    this.clustersKeyText = text;
    this.clustersList.replaceChildren(...ids.map(([id, names]) => this.keyItem('clusters', id, clusterColorCSS(id), compare('clusterKey', { n: number(id + 1) }),
      compare('clusterKeyTooltip', { n: number(id + 1), where: names.join(', ') }))));
  }
  /** The menu of Add: the run (when it is not a column) and the sample's segmentations that are not, each with its key,
   *  what made it and its cells; 4 columns already, or none left, says so. */
  renderMenu() {
    const full = this.keys.length >= MAX_COLUMNS, list = this.list;
    const items = full ? [] : [...(this.keys.includes('run') || !this.runId ? [] : [{ key: 'run' }]), ...list.items.filter(s => !this.keys.includes(s.key))];
    const note = full ? T.addFull : list.state === 'loading' ? T.addLoading : list.state === 'failed' ? T.addFailed : !items.length ? T.addNone : '';
    const mKey = JSON.stringify([items.map(s => s.key), full, list.state, this.iteration]);
    if (mKey !== this.menuKey) {
      this.menuKey = mKey;
      this.menuItems.replaceChildren(...items.map(s => {
        const b = this.el('button', 'cmp-menu-item'); b.type = 'button'; b.setAttribute('role', 'menuitem'); b.dataset.key = s.key;
        const { name, sub } = this.names(s.key, null), shapes = s.count != null ? compare('outlines', { n: number(s.count) }) : '';
        const line = [sub, shapes].filter(Boolean).join(T.sep), lineNode = this.el('span', 'cmp-menu-sub', line);
        lineNode.title = line;   // cut short in a narrow menu: the whole of it on hover
        b.append(this.el('span', 'cmp-menu-name', name), lineNode);
        b.onclick = event => { event?.stopPropagation?.(); this.add(s.key); };
        return b;
      }));
    }
    setText(this.menuNote, note); setHidden(this.menuNote, !note); setHidden(this.menuRetry, list.state !== 'failed' || full);
  }
  /** The hover's tip: the dot's kind (its swatch and genes) and its share from neighbours. */
  renderTip() {
    for (const col of this.cols.values()) if (!this.hover || col.key !== this.hover.key) setHidden(col.part.tip, true);
    const h = this.hover, col = h ? this.cols.get(h.key) : null;
    if (!col?.points) return;
    const p = col.part, kind = col.points.kind[h.index], code = col.points.share[h.index], kinds = this.status?.kinds;   // the same in its own map
    const k = kinds?.find(x => x.index === kind) ?? null;
    setText(p.tipName, k ? k.name || T.noKind : T.noKind); setHidden(p.tipName.parentElement, !kinds);   // no kinds learned: the share alone
    p.tipSwatch.style.background = k ? k.color : COLORS.none;
    const share = cellShare(code);
    const line = share == null ? T.mixUnjudged : compare('hoverShare', { p: sharePercent(share) }) + (stepOf(code) === 2 ? `${T.sep}${T.hoverMixed}` : '');
    setText(p.tipShare, line);
    // Coloured by clusters: its own cluster; by differences: its column's share of cells there beside the yardstick's.
    const by = this.colouring(), own = !!col.comp?.own;
    let more = '';
    if (by === 'clusters' && own && col.clusters && col.clusters[h.index] < NO_CLUSTER) more = compare('clusterTip', { n: number(col.clusters[h.index] + 1) });
    else if (by === 'differences' && !own && col.key !== this.yardstick()) {
      const r = this.diffOf(col)?.ratio[h.index];
      if (finite(r)) more = compare('diffTip', { r: number(Math.max(0.1, r), r >= 10 ? 0 : 1), ref: this.refName(true) });
    }
    setText(p.tipMore, more); setHidden(p.tipMore, !more);
    setHidden(p.tip, false); this.placeTip();
  }
  /** The tip beside the pointer, on the side with room (its own size measured: four columns make narrow plots), and
   *  kept inside the plot, which clips it. */
  placeTip() {
    const h = this.hover, col = h ? this.cols.get(h.key) : null;
    if (!col) return;
    const w = col.css?.[0] ?? col.part.plot.clientWidth ?? PLOT_PX[0], h0 = col.css?.[1] ?? col.part.plot.clientHeight ?? PLOT_PX[1];
    const tip = col.part.tip, tw = tip.offsetWidth || 0, th = tip.offsetHeight || 0, gap = 14, edge = 4;
    const side = (at, size, room) => at + gap + size <= room - edge || at - gap - size < edge ? at + gap : at - gap - size;
    const inside = (at, size, room) => Math.max(edge, Math.min(at, room - size - edge));
    const left = `${Math.round(inside(side(h.x, tw, w), tw, w))}px`, top = `${Math.round(inside(side(h.y, th, h0), th, h0))}px`;
    const s = tip.style;
    if (s.left !== left) s.left = left; if (s.top !== top) s.top = top;
    if (s.right !== 'auto') s.right = 'auto'; if (s.bottom !== 'auto') s.bottom = 'auto';
  }

  // ---- painting -----------------------------------------------------------------------------------------------------------
  painting() { return this.isOpen && !this.hidden; }
  /** A column's picture is about to change: it cross-fades from what is on its canvas (not under reduced or calm motion). */
  startFade(col) {
    col.dirty = true;
    col.fade = this.still ? null : { from: this.snapshotCanvas(col), start: this.now() };   // from: null, the background
  }
  snapshotCanvas(col) {
    const src = col.part.canvas;
    if (!(src.width > 0 && src.height > 0) || !col.painted) return null;
    const c = col.fadeCanvas?.canvas ?? this.doc.createElement('canvas');
    c.width = src.width; c.height = src.height;
    const ctx = c.getContext?.('2d');
    if (!ctx) return null;
    ctx.globalAlpha = 1; ctx.drawImage(src, 0, 0);
    col.fadeCanvas = { canvas: c };
    return c;
  }
  /** One paint is due: at once for a change, else at the fade's next tick; never sooner than TICK_MS after the last
   *  paint while a fade runs. */
  wake() {
    if (!this.painting() || this.paintFrame != null || this.timers.has('paint')) return;
    let fading = false, dirty = false, flying = false;
    for (const col of this.cols.values()) { fading ||= !!col.fade; dirty ||= col.dirty; flying ||= !!col.flight; }
    if (!fading && !dirty && !flying) return;
    const wait = (fading || flying) && !dirty ? this.paintedAt + (flying ? FLY_TICK_MS : TICK_MS) - this.now() : 0;
    this.later('paint', wait, () => {
      this.paintFrame = this.clock.requestAnimationFrame(() => { this.paintFrame = null; this.paint(); this.wake(); });
    });
  }
  /** Each column that changed or fades is painted: its picture (over the one it fades from), the hover's rings. */
  paint() {
    if (!this.painting()) return;
    const now = this.now(), still = this.still;
    for (const key of this.keys) {
      const col = this.cols.get(key);
      if (!col || (!col.dirty && !col.fade && !col.flight)) continue;
      col.dirty = false;
      this.paintColumn(col, now, still);
      this.draws++; this.hook?.count('compareDraws');
    }
    this.paintedAt = now;
  }
  /** The canvas at its plot's size in device px (a new size clears it: drawn again in full). */
  fit(col) {
    const c = col.part.canvas, css = col.css ?? [col.part.plot.clientWidth || PLOT_PX[0], col.part.plot.clientHeight || PLOT_PX[1]];
    const w = Math.max(1, Math.round(css[0] * this.dpr)), h = Math.max(1, Math.round(css[1] * this.dpr));
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    return { w, h, css };
  }
  paintColumn(col, now, still) {
    const canvas = col.part.canvas, size = this.fit(col), ctx = canvas.getContext?.('2d', { alpha: false });
    if (!ctx) return;
    if (col.flight && this.flyColumn(col, ctx, size, now, still)) return;
    const comp = this.composite(col, size), fade = col.fade, t = fade && !still ? fadeProgress(now - fade.start) : 1;
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = COLORS.background; ctx.fillRect(0, 0, size.w, size.h);
    if (fade && t < 1 && fade.from) ctx.drawImage(fade.from, 0, 0, size.w, size.h);
    if (comp) { ctx.globalAlpha = fade && t < 1 ? t : 1; ctx.drawImage(comp, 0, 0); }
    else if (col.ghost) { ctx.globalAlpha = GHOST_ALPHA; ctx.drawImage(col.ghost, 0, 0, size.w, size.h); }
    ctx.globalAlpha = 1;
    if (fade && t >= 1) col.fade = null;
    if (comp) this.rings(ctx, col);
    col.painted = true;
  }
  /** A flight's paint: each dot `ease(elapsed / FLY_MS)` of the way from its place on one map to its place on the other,
   *  in the colours it lands in. False once it has landed (the column is then drawn as it rests). */
  flyColumn(col, ctx, size, now, still) {
    const f = col.flight, t = still ? 1 : clamp((now - f.start + FLY_TICK_MS) / FLY_MS, 0, 1);
    if (!(col.img?.width === size.w && col.img?.height === size.h)) col.img = typeof ctx.createImageData === 'function' ? ctx.createImageData(size.w, size.h) : null;
    if (t >= 1 || !col.img?.data) { col.flight = null; return false; }
    const pad = PAD_PX * this.dpr, pa = placement(f.from.bounds, size.w, size.h, pad), pb = placement(f.to.bounds, size.w, size.h, pad);
    f.xy = flyPoints(f.from.pts, pa, f.to.pts, pb, ease(t), f.xy);
    const bg = rgbOf(COLORS.background), px = col.img.data;
    for (let o = 0; o < px.length; o += 4) { px[o] = bg[0]; px[o + 1] = bg[1]; px[o + 2] = bg[2]; px[o + 3] = 255; }
    const r = dotRadius(this.layout?.most ?? f.to.pts.n, size.css[0], size.css[1]) * this.dpr;
    rasterise(px, size.w, size.h, f.xy, FLY_PLACE, f.slots, f.palette, { r });
    ctx.globalAlpha = 1; ctx.putImageData(col.img, 0, 0);
    col.painted = true;
    return true;
  }
  /** The hover's rings: around the dot named in its column, and a fainter one at the same place in every other drawn on
   *  the same map. */
  rings(ctx, col) {
    const h = this.hover, from = h ? this.cols.get(h.key) : null, place = col.comp?.place, pts = from?.comp?.pts;
    if (!h || !pts || !place || from.layoutId !== col.layoutId || from.comp.own !== col.comp.own) return;
    const x = place.ox + place.k * pts.x[h.index], y = place.oy - place.k * pts.y[h.index];
    if (!finite(x) || !finite(y)) return;
    const own = from === col, r = col.comp.r + (own ? 3 : 5) * this.dpr;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.lineWidth = (own ? 1.5 : 1) * this.dpr; ctx.strokeStyle = own ? COLORS.ring : COLORS.echo; ctx.stroke();
  }
  /** A column's points drawn once at this size, map, colouring and lit kind, and kept (the paints lay it on the canvas). */
  composite(col, size) {
    const lay = this.layout, view = this.shown(col);
    if (!view) return null;
    const by = this.colouring(), lit = this.lit && this.lit.by === by ? this.lit : null, kinds = this.status?.kinds ?? null;
    const key = JSON.stringify([lay.id, view.own, by, lit ? lit.value : null, size.w, size.h, this.dpr, kinds?.map(k => [k.index, k.color]) ?? null]);
    if (col.comp?.key === key) return col.comp.canvas;
    const canvas = col.comp?.canvas ?? this.doc.createElement('canvas');
    if (canvas.width !== size.w) canvas.width = size.w;
    if (canvas.height !== size.h) canvas.height = size.h;
    const ctx = canvas.getContext?.('2d');
    if (!ctx) return null;
    const place = placement(view.bounds, size.w, size.h, PAD_PX * this.dpr), pts = view.pts;
    const r = dotRadius(lay.most, size.css[0], size.css[1]) * this.dpr, { slots, palette } = this.slotsOf(col, view, by, palettes(kinds));
    let flags = null;
    if (lit) { flags = new Uint8Array(pts.n); for (let i = 0; i < pts.n; i++) flags[i] = slots[i] === lit.value ? 1 : 0; }
    // The pixels: the column's own buffer, kept while the canvas keeps its size (no new 4 MB on every lit kind).
    if (!(col.img?.width === size.w && col.img?.height === size.h)) col.img = typeof ctx.createImageData === 'function' ? ctx.createImageData(size.w, size.h) : null;
    const img = col.img;
    let drawn = 0;
    if (img?.data) {
      const bg = rgbOf(COLORS.background), px = img.data;
      for (let o = 0; o < px.length; o += 4) { px[o] = bg[0]; px[o + 1] = bg[1]; px[o + 2] = bg[2]; px[o + 3] = 255; }
      drawn = rasterise(px, size.w, size.h, pts, place, slots, palette, { r, lit: flags });
      ctx.putImageData(img, 0, 0);
    } else {   // a canvas without pixel access: square dots, one fill per colour
      ctx.fillStyle = COLORS.background; ctx.fillRect(0, 0, size.w, size.h);
      const groups = new Map();
      for (let i = 0; i < pts.n; i++) { const g = groups.get(slots[i]) ?? []; g.push(i); groups.set(slots[i], g); }
      for (const [slot, list] of groups) {
        const c = palette.subarray(slot * 3, slot * 3 + 3);
        ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.globalAlpha = flags ? DIM_ALPHA : DOT_ALPHA;
        for (const i of list) {
          const x = place.ox + place.k * pts.x[i], y = place.oy - place.k * pts.y[i];
          if (x >= 0 && y >= 0 && x < size.w && y < size.h) { ctx.fillRect(x - r, y - r, 2 * r, 2 * r); drawn++; }
        }
      }
      ctx.globalAlpha = 1;
    }
    col.comp = { canvas, key, place, r, drawn, pts, own: view.own }; col.grid = null;
    return canvas;
  }

  // ---- the harness ------------------------------------------------------------------------------------------------------
  /** For the harness: open, the state, each column's {key, state, count (the count it is read on, or null), mixed_per100,
   *  like100_per100, points (the UMAP's points drawn: those of the layout on screen), own (drawn on its own map)}, the area ({box, size_um, tiles} or null); and colourBy,
   *  map ('own' | 'one'), kinds (how many, null without), fading (columns cross-fading), flying (columns flying between
   *  the maps), draws, fetches (every request: statuses, points, starts), polls (status asks), downloads (points, own
   *  maps and clusters) and opened. */
  snapshot() {
    const open = this.isOpen, s = open ? this.status : null;
    return {
      open, state: open ? this.state : 'closed',
      columns: open ? this.keys.map(key => {
        const col = this.cols.get(key), d = col?.data, view = col ? this.shown(col) : null;
        return { key, state: d?.state ?? (this.state === 'starting' ? 'starting' : this.state), count: d?.count ?? null, form: d?.form ?? null,
          like_from: d?.likeFrom ?? null, mixed_per100: d?.mixed ?? null,
          like100_per100: d?.like100 ?? null, points: view ? view.pts.n : 0, own: !!view?.own };
      }) : [],
      area: s?.area ? { box: s.area.box, size_um: s.area.size_um, tiles: s.area.tiles } : null,
      colourBy: open ? this.colouring() : null, map: open ? this.mapMode : null, kinds: s?.kinds ? s.kinds.length : null,
      fading: open ? [...this.cols.values()].filter(c => c.fade).length : 0, flying: open ? [...this.cols.values()].filter(c => c.flight).length : 0,
      draws: this.draws, fetches: this.fetches, polls: this.polls, downloads: this.downloads, opened: this.opened,
    };
  }
  /** Every word the view shows now (text, tooltips and labels), for the harness's banned-word check. */
  texts() {
    if (!this.isOpen) return [];
    const out = [];
    const walk = node => {
      if (!node || node.hidden) return;
      if (node.tagName === '#TEXT' || node.nodeType === 3) { const t = String(node.textContent ?? '').trim(); if (t) out.push(t); return; }
      for (const attr of ['title', 'aria-label']) { const v = node.getAttribute?.(attr); if (v) out.push(v); }
      if (!node.childNodes?.length && node.textContent && !node.children?.length) out.push(String(node.textContent).trim());
      for (const child of node.childNodes ?? []) walk(child);
    };
    walk(this.root);
    return [...new Set(out.filter(Boolean))];
  }
  /** Every string the view can show (its deck's; tests). */
  static allTexts() { return allTexts(); }
  dispose() {
    this.close(false);
    for (const key of [...this.timers.keys()]) this.cancel(key);
    for (const [node, type, fn, opts] of this.listeners) node?.removeEventListener?.(type, fn, opts);
    this.listeners = []; this.observer?.disconnect?.(); this.root.remove?.();
  }
}

function setText(node, text) { text = text == null ? '' : String(text); if (node && node.textContent !== text) node.textContent = text; }
function setHidden(node, hidden) { if (node && node.hidden !== hidden) node.hidden = hidden; }
function setTitle(node, title) { title = title ?? ''; if (node && node.title !== title) node.title = title; }
function setData(node, key, value) { if (node && node.dataset[key] !== value) node.dataset[key] = value; }
