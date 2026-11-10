// The test corner: a floating view of a test of a setting inside the stage, docked at its bottom left, moved by its
// header, made smaller to a strip. It shows what the panel's test state machine says (qc-panel.js builds the model:
// testModel lists its fields) and hands every press back as an action (onAction): it never starts, cancels or answers a
// test itself. The form holds one setting and its value, then the panel's lines (the nearby cells with the same pattern
// the engine found, the area its runs re-run, the runs) and why a test can't run here (the engine's refusals, in two
// groups); what a press would wait for first and where the test was not measured are Run's tooltip, never on screen. A
// test that waits for them (the run's last iteration, the mixing check, another run) says for what in one line, with
// Cancel.
//
// While the test's runs work (fresh runs of the area as now and with the value, 3 or 4 of each, all at once) it shows
// the first of each side (the engine's A1 and B1: the control and test views here) and asks for their newest frames
// (GET /runs/{id}/qc/test/view through fetchView) at most about three times a second, one ask at a time, and once more
// when the answer comes (the engine keeps each run's last view after deleting the runs): that one until a view that
// says the test ended comes, an ask that fails made again a little later, a few times (RETRY_MS, FINAL_TRIES), then
// the views say they may be an earlier iteration's; never while closed, before the test exists or while the page is
// hidden. Every distinct frame (by run and frame number) is kept for the scrubber, which replays them after the answer.
// A frame is decoded once. The progress under the views is the engine's: the slowest run of each side, a bar each, the
// schedule's parts marked on them.
//
// Each view is a 2-D canvas of the square crop around the cell (no WebGL context), drawn at the run's own grid (2 µm while
// settling, 1 µm after) placed by its own origin, as the stage draws cells: every cell faintly filled in its colour
// (the stage's colour of the cell it is in your run, colorOf, when the view names it; else a colour of its own) with its
// outline on its own side of the line, the neighbours (matched by name in that run) outlined magenta, the cell outlined
// white with a soft glow; the outlines are the drawing, never the count. Each suspect has the share of it proseg counts
// for the cell in that run's newest counted iteration (runs[k].held, 0 to 1: the engine's, exact with proseg's export
// of the view, else approximate, said in the count's tooltip): one it still counts glows (a bright core and a breathing
// halo, as bright as its likelihood times that share), one it counts under MOVED_BELOW of is a small teal ring. Under
// each view "Suspects kept: a of n": a the shares added up (rounded), n the suspects with a share, and proseg's count of
// the cell. A suspect the engine could not tell apart from another transcript of its gene at its place
// (runs[k].unsure) is a plain small grey ring, left out of every count and said in one muted line.
//
// The answer, part by part in the panel's words: do they come back, are they fewer (how often the test is wrong in its
// ⓘ, then where it was not measured), what it cost, where the suspects went (a bar per side), this cell alone, or too few
// to judge; then New run with this, Keep my setting and Run again, none of them the main one. The views keep their size
// at the answer: on a stage wide enough (WIDE_STAGE) it reads beside them, a column of its own past a hairline, else
// under them, and the card's body scrolls where the stage is too short for all of it.
//
// Motion: the halos breathe between 60 and 100 % over 1.6 s and a new frame fades in over 120 ms, both
// on one tick of at most 12 paints a second, only while the corner is open and not made smaller and the page is visible.
// Each frame's drawing (with the suspects' cores) and its halos are drawn once and kept; a tick that only breathes
// repaints just the glowing patch, from the kept drawing and halos at the breath's level (two image draws however many
// suspects). Reduced or calm motion: a still glow and no fade, one paint per change. Closed, made smaller or hidden:
// nothing is scheduled. No CSS motion of any kind (qc-test-corner.css).
import { CORNER_TEXT as T, corner, allTexts } from './qc-test-corner-copy.mjs';
import { number, settingValue, TEST_REASON_GROUPS, testReasonGroup } from './qc-copy.mjs';
import { attachSettingDemo } from './setting-demo.mjs';

// Sizes (CSS px; qc-test-corner.css holds the same numbers): a view's side (at least 220, also at the answer), the
// gap between the views, the card's padding and borders with the views' borders (14 + 14 + 1 + 1 + 2 x 2); the answer's
// column beside the views on a wide stage and the gap before it (14, a hairline, 14). The corner is 2 x 220 + 12 + 34 =
// 486 wide while watching and with the answer under the views (a narrow stage), 486 + 29 + 268 = 783 with the answer
// beside them, which takes a stage of 783 + 2 x 12 = 807 (WIDE_STAGE); 360 for the form and a stopped test.
export const VIEW_PX = 220, DONE_VIEW_PX = VIEW_PX, MIN_VIEW_PX = 120, VIEW_GAP = 12, FRAME_PX = 34;
export const ANSWER_PX = 268, ANSWER_GAP = 29;
export const WIDTH = 2 * VIEW_PX + VIEW_GAP + FRAME_PX, DONE_WIDTH = WIDTH, WIDE_WIDTH = WIDTH + ANSWER_GAP + ANSWER_PX;
export const FORM_WIDTH = 360, SMALL_WIDTH = 320, MARGIN = 12, WIDE_STAGE = WIDE_WIDTH + 2 * MARGIN;
// Everything but the views' side, top to bottom (CSS px, worked out from the stylesheet's line heights for a not-clear
// answer with every part, the longest verdict lines and the three answers; a browser measures it again): while watching
// (the views' titles and counts, the progress bars, the progress and runs lines with Cancel, the key), with the answer
// under the views, and with the answer beside them (the views' column: their titles and numbers, the scrubber, the key);
// `line` each muted line under the views when it shows (the suspects that couldn't be told apart, the views that may be
// older, the runs started again). A short stage gets smaller views while watching rather than a corner that scrolls; at
// the answer the views keep VIEW_PX and the card's body scrolls.
export const CHROME_PX = Object.freeze({ watch: 238, done: 665, wide: 205, line: 22 });
// The corner's size by phase until the browser reports it (ResizeObserver): [width, height], without the muted lines. With
// the answer beside the views the answer's column sets the card's height (its parts at 268 px wrap more); a test that
// waits is as small as one that stopped (a line and its button).
export const SIZES = Object.freeze({ form: [FORM_WIDTH, 300], watch: [WIDTH, 458], done: [WIDTH, 885], wide: [WIDE_WIDTH, 603],
  waiting: [FORM_WIDTH, 138], ended: [FORM_WIDTH, 138], small: [SMALL_WIDTH, 55] });
export const VIEW_MS = 350;            // the view is asked for at most this often (under 3 a second: qcTestViewFetches)
export const RETRY_MS = 1000;          // the views at the answer: a failed ask made again 1 s later, then 2 s, then 4 s,
export const FINAL_TRIES = 4;          // 4 asks in all; then the views say they may be an earlier iteration's
export const GLOW_FPS = 12;            // the glow and the fades: at most this many paints a second
export const TICK_MS = Math.ceil(1000 / GLOW_FPS);
export const BREATH_MS = 1600, BREATH_LOW = 0.6;    // the halos breathe between 60 and 100 % over 1.6 s
export const FADE_MS = 120;            // a new frame fades in over this long
export const REPLAY_MS = 150;          // Replay steps through the kept frames this fast
export const PENDING_MS = 3000;        // a pressed button waits this long for the panel's answer before it can send again
// proseg's default schedule (engine/qc_test.py schedule_total): its end, and where settling and refining end (the
// progress bars' marks until the engine's own come with the test).
export const OF = 450, MARKS = Object.freeze([200, 400]);
export const GLOW_MIN = 0.2;           // a suspect's likelihood glows from 0.2, brighter up to 1
export const MOVED_BELOW = 0.2;        // a suspect proseg counts under this share of for the cell: a teal ring, no glow
export const HALO_ALPHA = 0.9;         // a halo's strength at likelihood 1 and the top of a breath
export const HALO = 6;                 // a halo is about 6 times the core across (as the stage's: glow.mjs GLOW_HALO)
export const CROWD = 24;               // up to this many glowing suspects in a view each glows in full
export const MAX_FRAMES = 400;         // per run (the views come at most 3 a second; past this the second oldest goes)
export const MAX_COMPOSITES = 8;       // drawn frames kept for repaints (the two shown, the two fading out, the scrubber's)
export const MAX_PIXELS = 1 << 20;     // a crop larger than this is refused (140 µm at 1 µm is 19,600)
export const COLORS = Object.freeze({
  background: '#0b0e13',
  // By class (1 another cell, 2 a neighbour the cell's RNA seems to come from, 3 the cell): how strong a cell's own
  // colour fills it (faint, as the stage fills cells: the glow must lead), and its outline's colour (widths: LINES).
  fill: Object.freeze([0, 0.22, 0.26, 0.2]),
  edge: Object.freeze(['', '', '#d55181', '#ffffff']),   // another cell: its own colour, lighter (EDGE_LIFT, EDGE_ALPHA)
  cellGlow: 'rgba(255,255,255,0.7)',
  glow: '#ff4fa3', moved: '#2ec4b6',
  unsure: '#9aa4b2',   // a suspect that couldn't be told apart: a plain grey ring, smaller and thinner than a teal one
  // The suspects' glow, as the stage's (glow.mjs): a pink halo and a white-pink core.
  halo: Object.freeze([[0, 'rgba(255,79,163,1)'], [0.18, 'rgba(255,79,163,0.8)'], [0.42, 'rgba(255,79,163,0.38)'], [0.7, 'rgba(255,79,163,0.1)'], [1, 'rgba(255,79,163,0)']]),
  core: Object.freeze([[0, 'rgba(255,244,250,1)'], [0.5, 'rgba(255,190,222,1)'], [0.8, 'rgba(255,79,163,0.85)'], [1, 'rgba(255,79,163,0)']]),
});
const LINES = Object.freeze([0, 1, 1.5, 2]);   // outline widths by class, CSS px
const EDGE_LIFT = 0.35, EDGE_ALPHA = 0.55;     // another cell's outline: its colour 35 % toward white (stage.js), at 55 %
const CELL_GLOW_PX = 6;                        // the soft glow around the cell's white outline (CSS px)
const CORE_SPAN = 1.3;                         // a core's gradient reaches this far past its bright middle
export const COUNTERS = Object.freeze(['qcTestViewFetches', 'qcTestCornerDraws']);
/** The views' slots: one run as now (the engine's side A, its first run), one with the value (side B). */
export const ROLES = Object.freeze(['control', 'test']);
export const SIDES = Object.freeze({ control: 'A', test: 'B' });
export const PHASES = Object.freeze(['form', 'starting', 'waiting', 'preparing', 'running', 'comparing', 'done', 'failed', 'cancelled']);
export const ACTIONS = Object.freeze(['setting', 'value', 'run', 'cancel', 'new', 'keep', 'again', 'close']);
/** The answer's parts, in their order (the panel's testReadout), and the went bars' parts. */
export const PARTS = Object.freeze(['back', 'fewer', 'cost', 'went', 'alone', 'toofew']);
export const WENT = Object.freeze(['stay', 'neighbour', 'other', 'background']);
export const COST_ROWS = Object.freeze(['own', 'all', 'perCell']);
/** The refusals' groups (qc-copy.mjs TEST_REASON_GROUPS): each one's heading and what it means (its tooltip). */
const REASON_HEADS = Object.freeze({ first: [T.reasonsFirst, T.reasonsFirstTooltip], cant: [T.reasonsCant, T.reasonsCantTooltip] });
const FORM_ACTIONS = new Set(['setting', 'value']);   // sent as they come, never held back
const NEW_TEST = new Set(['starting', 'waiting', 'preparing', 'running', 'comparing']);   // entered from elsewhere: another test
const ASKING = new Set(['preparing', 'running', 'comparing']);
const VIEWING = new Set(['starting', 'preparing', 'running', 'comparing', 'done']);
const ENDED = new Set(['failed', 'cancelled']);
const VIEW_ENDED = new Set(['done', 'failed', 'cancelled']);
const RUNS_ENDED = new Set(['comparing', ...VIEW_ENDED]);   // the test's runs ended (comparing: every one done)
const CLOSABLE = new Set(['form', 'done', 'failed', 'cancelled']);   // a test on its way stops with Cancel only
const finite = Number.isFinite;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const LITTLE = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

// Icons: one stroke weight, the text colour.
const svg = body => `<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
export const ICONS = Object.freeze({
  info: svg('<circle cx="8" cy="8" r="6.25"/><path d="M8 7.25v4M8 4.75v.01"/>'),
  minimise: svg('<path d="M4 11.5h8"/>'),
  restore: svg('<rect x="3.75" y="3.75" width="8.5" height="8.5" rx="1.5"/>'),
  close: svg('<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>'),
  play: svg('<path d="M5.5 3.75l6.5 4.25-6.5 4.25z" fill="currentColor"/>'),
  stop: svg('<rect x="4.5" y="4.5" width="7" height="7" rx="1" fill="currentColor"/>'),
});

// ---- pure helpers (node tests) ------------------------------------------------------------------------------------
/** The view's labels (base64 of little-endian u32, `n` of them) as a Uint32Array; null when they are not that. */
export function decodeLabels(b64, n) {
  if (typeof b64 !== 'string' || !Number.isInteger(n) || n <= 0 || n > MAX_PIXELS) return null;
  let bytes;
  try {
    if (typeof globalThis.atob === 'function') {
      const s = globalThis.atob(b64); bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    } else bytes = Uint8Array.from(globalThis.Buffer.from(b64, 'base64'));
  } catch { return null; }
  if (bytes.length !== n * 4) return null;
  if (LITTLE) return new Uint32Array(bytes.buffer, 0, n);
  const out = new Uint32Array(n), view = new DataView(bytes.buffer);
  for (let i = 0; i < n; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}

const labelOf = v => (Number.isInteger(v) && v > 0 ? v : null);
/** The view slot of a run of the view reply: the engine's side ("A": as now, "B": with the value; else the side its role
 *  names, "A1", "B1"), as the control and test views here; an older engine's role ("control", "test") as it is. */
function slotOf(r) {
  const side = r.side === 'A' || r.side === 'B' ? r.side : typeof r.role === 'string' && /^[AB]\d+$/.test(r.role) ? r.role[0] : null;
  return side === 'A' ? 'control' : side === 'B' ? 'test' : ROLES.includes(r.role) ? r.role : null;
}
/** One run of the view reply checked (its labels still encoded); null when it is not one. `n`: the suspects. `held`:
 *  per suspect the share of it proseg counts for the cell in that frame (a number from 0 to 1, clamped), NaN where the
 *  engine knows none (null: its transcript not found or not told apart, or no cell); a list that does not line up with
 *  the suspects knows none. `unsure` (1 where the engine could not tell the suspect from another transcript of its gene
 *  at its place: never counted, engine/qc_test.py match) the same; without it none is. `heldBy`: "exact" (proseg's
 *  export of the view) or "approx". `parents` (when the engine names them: flat pairs [label, the label of the cell of
 *  that name in your run, ...]) becomes a Map; null without it. `role` is its view's slot (slotOf), `run` the engine's
 *  own role ("A1", "B1"). */
function normaliseRun(r, n) {
  const role = r && typeof r === 'object' ? slotOf(r) : null;
  if (!role) return null;
  if (!Number.isInteger(r.seq) || r.seq < 0 || !Number.isInteger(r.w) || !Number.isInteger(r.h) || r.w <= 0 || r.h <= 0) return null;
  if (r.w * r.h > MAX_PIXELS || !(finite(r.voxel) && r.voxel > 0) || !finite(r.x0) || !finite(r.y0) || typeof r.labels !== 'string') return null;
  const held = new Float64Array(n).fill(NaN), unsure = new Uint8Array(n);
  if (Array.isArray(r.unsure) && r.unsure.length === n) r.unsure.forEach((v, k) => { unsure[k] = v ? 1 : 0; });
  // The engine never gives an unsure one a share; were both said, it stays out (drawn and counted as unsure).
  if (Array.isArray(r.held) && r.held.length === n) r.held.forEach((v, k) => { if (finite(v) && !unsure[k]) held[k] = clamp(v, 0, 1); });
  let parents = null;
  if (Array.isArray(r.parents)) {
    parents = new Map();
    for (let i = 0; i + 1 < r.parents.length; i += 2) {
      const a = labelOf(r.parents[i]), b = labelOf(r.parents[i + 1]);
      if (a != null && b != null) parents.set(a, b);
    }
  }
  return { role, run: typeof r.role === 'string' ? r.role : null, seq: r.seq, iter: finite(r.iter) ? r.iter : null, voxel: r.voxel, x0: r.x0, y0: r.y0,
    w: r.w, h: r.h, labels: r.labels, cell: labelOf(r.cell), sources: (Array.isArray(r.sources) ? r.sources : []).map(labelOf),
    held, heldBy: r.held_by === 'exact' || r.held_by === 'approx' ? r.held_by : null, unsure,
    cellTotal: finite(r.cell_total) && r.cell_total >= 0 ? r.cell_total : null, parents };
}
/** A frame's suspects as drawn and counted: `weights` (each glowing one's share, MOVED_BELOW or more; 0 for the rest),
 *  `moved` (1 where the share is known and under MOVED_BELOW: a teal ring), `kept` (the shares known added up, rounded)
 *  and `of` (how many are known). Without the cell, nothing is (it is not in that run). */
export function shares(held, cell) {
  const n = held.length, weights = new Float64Array(n), moved = new Uint8Array(n);
  let sum = 0, of = 0;
  if (cell != null) {
    for (let k = 0; k < n; k++) {
      const v = held[k];
      if (!finite(v)) continue;
      of++; sum += v;
      if (v >= MOVED_BELOW) weights[k] = v; else moved[k] = 1;
    }
  }
  return { weights, moved, kept: Math.round(sum), of };
}
/** The test view (GET /runs/{id}/qc/test/view) checked and normalised; null when it is not one. Lenient about extra
 *  fields; a suspect without a place keeps its index (held is by index) and is never drawn. */
export function normaliseView(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const b = raw.box;
  const box = Array.isArray(b) && b.length === 4 && b.every(finite) && b[2] > b[0] && b[3] > b[1] ? b.map(Number) : null;
  const suspects = (Array.isArray(raw.suspects) ? raw.suspects : []).map(s => {
    const ok = !!s && finite(s.x) && finite(s.y);
    return { x: ok ? s.x : 0, y: ok ? s.y : 0, gene: typeof s?.gene === 'string' ? s.gene : '', p: finite(s?.p) ? clamp(s.p, 0, 1) : 0, ok };
  });
  const runs = (Array.isArray(raw.runs) ? raw.runs : []).map(r => normaliseRun(r, suspects.length)).filter(Boolean);
  return { state: typeof raw.state === 'string' ? raw.state : null, box, suspects, runs };
}

/** One frame's shapes at its grid (w x h, row-major, y down), cell by cell: `ids` the labels in it (one slot each, in
 *  the order met), `kinds` each slot's class (1 another cell, 2 a neighbour, 3 the cell), `fills` per slot its row runs
 *  [i0, j, i1, ...] and `edges` per slot its outline as segments [x0, y0, x1, y1, side, ...] (grid units) on the lines
 *  between it and anything else, `side` -1 when the cell lies before the line (left of it or above), 1 after it: drawn on
 *  its own side of the line, each cell's outline is its own, as the stage draws them. Runs along a row or column are
 *  merged. */
export function outline(labels, w, h, cell, sources = []) {
  const src = new Set((sources || []).filter(v => v != null && v !== cell));
  const slots = new Map(), ids = [], kinds = [], fills = [], edges = [];
  const slot = l => {
    let s = slots.get(l);
    if (s === undefined) { s = ids.length; slots.set(l, s); ids.push(l); kinds.push(l === cell ? 3 : src.has(l) ? 2 : 1); fills.push([]); edges.push([]); }
    return s;
  };
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w;) {
      const l = labels[j * w + i]; let e = i + 1;
      while (e < w && labels[j * w + e] === l) e++;
      if (l) fills[slot(l)].push(i, j, e);
      i = e;
    }
  }
  // Between columns i and i + 1 (x = i + 1), down the rows; between rows j and j + 1 (y = j + 1), along the columns: a
  // run holds one pair of cells (or a cell and the background).
  for (let i = 0; i < w - 1; i++) {
    for (let j = 0; j < h;) {
      const a = labels[j * w + i], b = labels[j * w + i + 1]; let e = j + 1;
      while (e < h && labels[e * w + i] === a && labels[e * w + i + 1] === b) e++;
      if (a !== b) { if (a) edges[slot(a)].push(i + 1, j, i + 1, e, -1); if (b) edges[slot(b)].push(i + 1, j, i + 1, e, 1); }
      j = e;
    }
  }
  for (let j = 0; j < h - 1; j++) {
    for (let i = 0; i < w;) {
      const a = labels[j * w + i], b = labels[(j + 1) * w + i]; let e = i + 1;
      while (e < w && labels[j * w + e] === a && labels[(j + 1) * w + e] === b) e++;
      if (a !== b) { if (a) edges[slot(a)].push(i, j + 1, e, j + 1, -1); if (b) edges[slot(b)].push(i, j + 1, e, j + 1, 1); }
      i = e;
    }
  }
  return { ids: Uint32Array.from(ids), kinds: Uint8Array.from(kinds), fills: fills.map(a => Int32Array.from(a)), edges: edges.map(a => Int32Array.from(a)) };
}

/** The stage's own colour of a cell (stage.js identityColor, main.js swatch: the colour when cells are not coloured by
 *  cluster) as [r, g, b], 0-255. */
export function identityRGB(id) {
  id >>>= 0;
  let h = Math.imul(id, 2654435761) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0; h = Math.imul(h, 2246822519) >>> 0; h = (h ^ (h >>> 13)) >>> 0;
  const hue = ((id * 0.6180339887) % 1 + (h & 255) / 1400) % 1;
  const sat = 0.52 + 0.33 * ((h >>> 8) & 255) / 255, val = 0.82 + 0.18 * ((h >>> 16) & 255) / 255;
  const k = n => val * (1 - sat + sat * clamp(Math.abs(((hue * 6 + n) % 6) - 3) - 1, 0, 1));
  return [k(0), k(4), k(2)].map(v => Math.round(v * 255));
}
/** A CSS colour as the page gives it ('#rrggbb' or 'rgb(r, g, b)') as [r, g, b], else null. */
export function parseColor(text) {
  if (typeof text !== 'string') return null;
  const s = text.trim(), hex = /^#([0-9a-f]{6})$/i.exec(s);
  if (hex) { const v = parseInt(hex[1], 16); return [v >> 16, (v >> 8) & 255, v & 255]; }
  const rgb = /^rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[,)]/i.exec(s);
  return rgb ? [rgb[1], rgb[2], rgb[3]].map(v => clamp(Math.round(Number(v)), 0, 255)) : null;
}
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const lift = (c, t) => c.map(v => Math.round(v + (255 - v) * t));

/** Where the crop `box` ([x0, y0, x1, y1] µm) lands in a square view of `size` device px: device px per µm (k), the box's
 *  offsets (centred), its clip and the glow's sizes (a core of about 0.9 µm, 1.75-3.5 CSS px; a halo HALO times it,
 *  24-44 CSS px across, as bright as the stage's; a teal ring just outside the core; a grey one the core's size). */
export function geometry(box, size, dpr = 1) {
  const bw = box[2] - box[0], bh = box[3] - box[1], k = size / Math.max(bw, bh);
  const ox = (size - bw * k) / 2, oy = (size - bh * k) / 2;
  const rc = clamp(0.9 * k, 1.75 * dpr, 3.5 * dpr), rh = clamp(HALO * rc, 12 * dpr, 22 * dpr);
  return { x0: box[0], y0: box[1], k, ox, oy, size, dpr, rc, rh, ring: rc + dpr, grey: rc,
    clip: [Math.round(ox), Math.round(oy), Math.round(ox + bw * k) - Math.round(ox), Math.round(oy + bh * k) - Math.round(oy)] };
}
const sx = (g, x) => g.ox + (x - g.x0) * g.k, sy = (g, y) => g.oy + (y - g.y0) * g.k;

/** The glow's level at `now` (ms): from 60 % up to 100 % and back, once every 1.6 s. */
export function breath(now) { return BREATH_LOW + (1 - BREATH_LOW) * (0.5 - 0.5 * Math.cos((2 * Math.PI * now) / BREATH_MS)); }
/** A fade's progress `elapsed` ms after it began: the first paint already moves one tick, the end at FADE_MS. */
export function fadeProgress(elapsed) { return clamp((elapsed + TICK_MS) / (FADE_MS + TICK_MS), 0, 1); }

/** Draws one frame of a run on `ctx` (a square of g.size device px): the background, then at the run's grid, placed by
 *  its own origin and clipped to the crop, every cell faintly filled in its colour (`colors[slot]`, [r, g, b]; else its
 *  own, identityRGB) and outlined on its own side of the line: another cell in its colour made lighter, a neighbour
 *  magenta, the cell white with a soft glow; then a teal ring on each suspect proseg no longer counts for the cell
 *  (`frame.moved`: a share under MOVED_BELOW), and a small grey ring on each the engine could not tell apart
 *  (`frame.unsure`: no share either way). Returns the teal rings. */
export function drawFrame(ctx, frame, g, suspects = [], colors = null) {
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = COLORS.background; ctx.fillRect(0, 0, g.size, g.size);
  const shape = frame?.shape;
  if (shape) {
    const X0 = sx(g, frame.x0), Y0 = sy(g, frame.y0), step = frame.voxel * g.k;
    const gx = i => Math.round(X0 + i * step), gy = j => Math.round(Y0 + j * step);
    const color = s => colors?.[s] ?? identityRGB(shape.ids[s]);
    ctx.save(); ctx.beginPath(); ctx.rect(...g.clip); ctx.clip();
    for (let s = 0; s < shape.ids.length; s++) {
      const runs = shape.fills[s]; if (!runs.length) continue;
      ctx.beginPath();
      for (let r = 0; r < runs.length; r += 3) { const x = gx(runs[r]), y = gy(runs[r + 1]); ctx.rect(x, y, gx(runs[r + 2]) - x, gy(runs[r + 1] + 1) - y); }
      ctx.fillStyle = rgba(color(s), COLORS.fill[shape.kinds[s]]); ctx.fill();
    }
    // The outlines, the cell's last (over its neighbours'); each line lies wholly on its cell's side, so two cells side
    // by side show both their outlines, every edge on device pixels.
    ctx.lineCap = 'butt';
    for (const kind of [1, 2, 3]) {
      const width = Math.max(1, Math.round(LINES[kind] * g.dpr));
      for (let s = 0; s < shape.ids.length; s++) {
        const segs = shape.edges[s];
        if (shape.kinds[s] !== kind || !segs.length) continue;
        ctx.beginPath();
        for (let e = 0; e < segs.length; e += 5) {
          const off = segs[e + 4] * width / 2;
          if (segs[e] === segs[e + 2]) { const x = gx(segs[e]) + off; ctx.moveTo(x, gy(segs[e + 1])); ctx.lineTo(x, gy(segs[e + 3])); }
          else { const y = gy(segs[e + 1]) + off; ctx.moveTo(gx(segs[e]), y); ctx.lineTo(gx(segs[e + 2]), y); }
        }
        ctx.lineWidth = width; ctx.strokeStyle = COLORS.edge[kind] || rgba(lift(color(s), EDGE_LIFT), EDGE_ALPHA);
        if (kind === 3) { ctx.shadowColor = COLORS.cellGlow; ctx.shadowBlur = CELL_GLOW_PX * g.dpr; }
        ctx.stroke();
      }
    }
    ctx.restore();
  }
  // A teal ring on each suspect proseg no longer counts for the cell (none when the cell is not in this run: nothing
  // left it); a smaller, thinner grey one on each that couldn't be told apart, whether the cell is there or not.
  const moved = frame?.cell != null ? frame.moved : null, unsure = frame?.unsure;
  let rings = 0, grey = 0;
  ctx.beginPath();
  suspects.forEach((s, k) => {
    if (!s.ok || !moved || !moved[k] || unsure?.[k]) return;
    const x = sx(g, s.x), y = sy(g, s.y);
    ctx.moveTo(x + g.ring, y); ctx.arc(x, y, g.ring, 0, 2 * Math.PI); rings++;
  });
  if (rings) { ctx.globalAlpha = 0.9; ctx.lineWidth = 1.25 * g.dpr; ctx.strokeStyle = COLORS.moved; ctx.stroke(); ctx.globalAlpha = 1; }
  ctx.beginPath();
  suspects.forEach((s, k) => {
    if (!s.ok || !unsure?.[k]) return;
    const x = sx(g, s.x), y = sy(g, s.y);
    ctx.moveTo(x + g.grey, y); ctx.arc(x, y, g.grey, 0, 2 * Math.PI); grey++;
  });
  if (grey) { ctx.globalAlpha = 0.85; ctx.lineWidth = g.dpr; ctx.strokeStyle = COLORS.unsure; ctx.stroke(); ctx.globalAlpha = 1; }
  return rings;
}

/** The suspects that glow, on screen: [x, y, p, weight] for each with a place and a weight above 0 (`weights[k]`: the
 *  share proseg still counts for the cell, 0 for one it no longer counts), p clamped to GLOW_MIN-1; and how crowded
 *  they are (1 up to CROWD, less past it). */
function glowing(suspects, weights, g) {
  const lit = [];
  suspects.forEach((s, k) => {
    const w = weights?.[k] ?? 0;
    if (s.ok && w > 0) lit.push([sx(g, s.x), sy(g, s.y), clamp(s.p, GLOW_MIN, 1), w]);
  });
  return { lit, crowd: clamp(Math.sqrt(CROWD / Math.max(1, lit.length)), 0.35, 1) };
}
/** The suspects' halos: each as strong as its likelihood times its weight times `level` (the breath), screened, not
 *  added, and softer past CROWD glowing suspects, so a crowd deepens to the glow's pink instead of burning out to a white
 *  patch (white is the cell's outline). Returns how many glow. */
export function drawGlow(ctx, suspects, weights, g, level, sprite = null) {
  const { lit, crowd } = glowing(suspects, weights, g);
  if (!lit.length) return 0;
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  for (const [x, y, p, w] of lit) {
    const a = HALO_ALPHA * p * w * level * (0.5 + 0.5 * crowd);
    if (sprite) { ctx.globalAlpha = a; ctx.drawImage(sprite, x - g.rh, y - g.rh, 2 * g.rh, 2 * g.rh); }
    else { ctx.globalAlpha = 0.35 * a; ctx.fillStyle = COLORS.glow; ctx.beginPath(); ctx.arc(x, y, g.rh, 0, 2 * Math.PI); ctx.fill(); }
  }
  ctx.restore();
  return lit.length;
}
/** The suspects' cores (white-pink, brighter than the halos), drawn with the frame and held still while the halos
 *  breathe, as the stage keeps a transcript's place plain; past CROWD glowing suspects they grow fainter and smaller, so
 *  a crowd reads as points in a pink field. Returns how many. */
export function drawCores(ctx, suspects, weights, g, sprite = null) {
  const { lit, crowd } = glowing(suspects, weights, g);
  if (!lit.length) return 0;
  const r = g.rc * (0.6 + 0.4 * crowd);
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  for (const [x, y, p, w] of lit) {
    ctx.globalAlpha = (0.55 + 0.45 * p) * w * crowd;
    if (sprite) ctx.drawImage(sprite, x - r * CORE_SPAN, y - r * CORE_SPAN, 2 * r * CORE_SPAN, 2 * r * CORE_SPAN);
    else { ctx.fillStyle = COLORS.core[0][1]; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
  }
  ctx.restore();
  return lit.length;
}
/** The patch a breath repaints: the halos of the glowing suspects (device px [x, y, w, h], inside the view), or null. */
export function glowBox(suspects, weights, g) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  suspects.forEach((s, k) => {
    if (!s.ok || !(weights?.[k] > 0)) return;
    const x = sx(g, s.x), y = sy(g, s.y);
    x0 = Math.min(x0, x - g.rh); y0 = Math.min(y0, y - g.rh); x1 = Math.max(x1, x + g.rh); y1 = Math.max(y1, y + g.rh);
  });
  if (!(x1 > x0)) return null;
  const a = clamp(Math.floor(x0) - 2, 0, g.size), b = clamp(Math.floor(y0) - 2, 0, g.size);
  const c = clamp(Math.ceil(x1) + 2, 0, g.size), d = clamp(Math.ceil(y1) + 2, 0, g.size);
  return c > a && d > b ? [a, b, c - a, d - b] : null;
}

// ---- the corner -------------------------------------------------------------------------------------------------------
/**
 * new TestCorner(host, options): the corner inside `host` (the stage's wrapper), hidden until show(). Options:
 * fetchView(runId) -> the test view (api.js qcTestView, as main.js passes it), onAction(action, payload) (ACTIONS:
 * 'setting' with the flag, 'value' with the text typed, and 'run', 'cancel', 'new', 'keep', 'again', 'close'),
 * onBox(box | null) (the Mixing legend moves out of its way: {left, top, bottom, w, h} CSS px in the stage),
 * colorOf(label) -> a CSS colour ('#rrggbb' or 'rgb(…)') of the cell of that label in the page's run, as the stage
 * colours it (the views colour each test cell by the cell of its name when the view names it; else, and without colorOf,
 * by the stage's own identity colour), reducedMotion and calmMotion (booleans or functions), hook (the harness:
 * COUNTERS), stage() -> {width, height} (else the host's size). Testing: clock {now, setTimeout, clearTimeout,
 * requestAnimationFrame, cancelAnimationFrame}.
 */
export class TestCorner {
  constructor(host, options = {}) {
    this.host = host; this.o = options; this.doc = host?.ownerDocument ?? globalThis.document;
    this.hook = options.hook ?? null;
    this.clock = options.clock ?? {
      now: () => globalThis.performance?.now() ?? Date.now(),
      setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms), clearTimeout: id => globalThis.clearTimeout(id),
      requestAnimationFrame: fn => globalThis.requestAnimationFrame(fn), cancelAnimationFrame: id => globalThis.cancelAnimationFrame(id),
    };
    this.listeners = []; this.timers = new Map(); this.paintFrame = null; this.sprites = {};
    this.isOpen = false; this.phase = null; this.model = null; this.runId = null; this.minimised = false; this.pending = null;
    this.left = MARGIN; this.bottom = MARGIN; this.size = null; this.boxKey = null; this.viewPx = VIEW_PX; this.dpr = 1;
    // The answer beside the views (layout); the muted lines that show under the views (unsure suspects, views that may be
    // older, the runs started again).
    this.beside = false; this.shownLines = { views: false, stale: false, retry: false };
    this.fetching = false; this.fetchedAt = -Infinity; this.paintedAt = -Infinity; this.glowTimes = [];
    this.draws = 0; this.fetches = 0; this.gen = 0;
    this.covered = false;   // a full-screen view over the page (setCovered): as while the page is hidden
    this.clearTest();
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
  get hidden() { return !!this.doc?.hidden || this.covered; }
  /** Reduced or calm motion: a still glow and no fades. */
  get still() {
    const flag = v => { try { return typeof v === 'function' ? !!v() : !!v; } catch { return false; } };
    if (flag(this.o.calmMotion)) return true;
    if (this.o.reducedMotion !== undefined) return flag(this.o.reducedMotion);
    try { return !!(this.media ??= globalThis.matchMedia?.('(prefers-reduced-motion: reduce)'))?.matches; } catch { return false; }
  }
  stageSize() {
    const s = this.call('stage');
    if (s && finite(s.width) && finite(s.height)) return s;
    return { width: this.host?.clientWidth || 0, height: this.host?.clientHeight || 0 };
  }
  tool(action, icon, label) {
    const b = this.el('button', 'qc-tc-tool'); b.type = 'button'; b.dataset.action = action; b.title = label; b.setAttribute('aria-label', label);
    b.innerHTML = icon;
    return b;
  }
  /** A button of the corner: never disabled; a press goes to the panel (onAction). */
  button(text, action, className = 'qc-button') {
    const b = this.el('button', className, text); b.type = 'button'; b.dataset.action = action;
    b.onclick = event => { event?.stopPropagation?.(); this.act(action); };
    return b;
  }
  /** An ⓘ that explains the line it sits on (its tooltip; focusable for the keyboard). */
  info() {
    const i = this.el('span', 'qc-tc-info'); i.tabIndex = 0; i.setAttribute('role', 'img'); i.innerHTML = ICONS.info;
    return i;
  }
  /** A muted line of the suspects the runs couldn't tell apart, led by their grey ring (hidden until some are). */
  unsureLine(tooltip) {
    const p = this.el('p', 'qc-tc-unsure'); p.title = tooltip; p.hidden = true;
    p.append(this.el('i'), this.el('span'));
    return p;
  }

  // ---- the DOM, built once ------------------------------------------------------------------------------------------
  build() {
    const el = (t, c, x) => this.el(t, c, x);
    const root = this.root = el('section', 'qc-tc');
    root.id = 'qc-test-corner'; root.hidden = true; root.tabIndex = -1;
    root.setAttribute('role', 'dialog'); root.setAttribute('aria-modal', 'false'); root.setAttribute('aria-labelledby', 'qc-tc-title');
    root.dataset.qcTestCorner = ''; root.dataset.qcCopy = ''; root.dataset.phase = 'closed';   // the harness's hooks and copy check
    // Header (the drag handle): the title and what is tested on which cell; Make smaller, Close.
    const head = this.head = el('header', 'qc-tc-head'); head.title = T.drag;
    const names = el('div', 'qc-tc-names');
    this.titleNode = el('h2', 'qc-tc-title'); this.titleNode.id = 'qc-tc-title';
    this.subNode = el('p', 'qc-tc-sub');
    names.append(this.titleNode, this.subNode);
    this.smallButton = this.tool('minimise', ICONS.minimise, T.minimise); this.smallButton.setAttribute('aria-pressed', 'false');
    this.closeButton = this.tool('close', ICONS.close, T.close); this.closeButton.title = T.closeTooltip;
    const tools = el('div', 'qc-tc-tools'); tools.append(this.smallButton, this.closeButton);
    head.append(names, tools);
    const body = this.body = el('div', 'qc-tc-body');
    // Choosing: the setting (each choice says whether a test tried it: data-tried) and its value beside the run's value
    // now, the setting's note (far drift's, what it does in its tooltip); the panel's lines (the nearby cells with the
    // same pattern, their numbers as data and what a test needs in their tooltip; the area; the runs, how they are made in
    // their tooltip) and why a test can't run here (the engine's refusals, each by its code, in their groups); Run the
    // test (never greyed out: what it still needs is listed under it, or what a press while refusals are listed says;
    // what a press would wait for first and where the test was not measured in its tooltip, their codes as data).
    const form = this.formBox = el('section', 'qc-tc-form');
    const field = (text, control, extra = null) => {
      const line = el('div', 'qc-tc-field'), label = el('label', 'qc-tc-label', text);
      label.htmlFor = control.id; line.append(label, control); if (extra) line.append(extra); return line;
    };
    this.select = el('select', 'qc-tc-select'); this.select.id = 'qc-tc-setting'; this.select.dataset.action = 'setting';
    const input = this.input = el('input', 'qc-tc-value'); input.id = 'qc-tc-value'; input.type = 'number'; input.step = 'any';
    input.dataset.action = 'value'; input.setAttribute('inputmode', 'decimal'); input.setAttribute('autocomplete', 'off');
    this.nowNode = el('span', 'qc-tc-now'); this.noteNode = el('p', 'qc-tc-note');
    const lines = el('div', 'qc-tc-lines');
    this.poolNode = el('p', 'qc-tc-pool'); this.areaNode = el('p', 'qc-tc-area'); this.runsNode = el('p', 'qc-tc-runs');
    for (const node of [this.poolNode, this.areaNode, this.runsNode]) node.hidden = true;
    lines.append(this.poolNode, this.areaNode, this.runsNode);
    this.reasons = el('div', 'qc-tc-reasons'); this.reasons.hidden = true; this.reasonsKey = null;
    this.runButton = this.button(T.run, 'run', 'qc-button qc-tc-primary'); this.runButton.title = T.runTooltip;
    this.problems = el('div', 'qc-tc-problems'); this.problems.setAttribute('role', 'alert'); this.problems.hidden = true;
    const formActions = el('div', 'qc-tc-actions'); formActions.append(this.runButton);
    // The "?" beside the choice: what the chosen setting does to a made-up cell, played in a popover beside the corner
    // (setting-demo.mjs; still under reduced or calm motion).
    this.demoButton = el('button', 'sdemo-q');
    const settingField = field(T.settingLabel, this.select, this.demoButton); settingField.classList.add('qc-tc-field-q');
    this.demo = attachSettingDemo(this.demoButton, { still: () => this.still, hook: this.hook, corner: () => this.root });
    // A setting offered as values to pick (the starting grid's, drift's on and off) shows them as buttons in the
    // value's place, the run's own one not to be picked; a typed number keeps the field.
    this.choiceBox = el('div', 'qc-tc-choice'); this.choiceBox.setAttribute('role', 'radiogroup');
    this.choiceBox.setAttribute('aria-label', T.valueChoiceLabel); this.choiceBox.hidden = true; this.choiceKey = null;
    const valueField = field(T.valueLabel, input, this.nowNode); valueField.insertBefore?.(this.choiceBox, this.nowNode);
    form.append(settingField, valueField, this.noteNode, lines, this.reasons, formActions, this.problems);
    // Waiting: what the test waits for first, in one line (the panel's words for the first; every one in its tooltip when
    // there are more, their codes as data), and Cancel. No views until its runs start.
    const waiting = this.waitBox = el('section', 'qc-tc-waiting-box');
    this.waitingLine = el('p', 'qc-tc-waiting'); this.waitingLine.setAttribute('role', 'status');
    this.waitCancel = this.button(T.cancel, 'cancel'); this.waitCancel.title = T.cancelTooltip;
    const waitActions = el('div', 'qc-tc-actions'); waitActions.append(this.waitCancel);
    waiting.append(this.waitingLine, waitActions);
    // Watching: the two views (as now, with the value: data-side A and B) and their numbers; the progress (a thin bar
    // per side, the schedule's parts marked; its line and Cancel; the runs; the runs started again) or, after the
    // answer, the scrubber; the key.
    const watch = this.watchBox = el('section', 'qc-tc-watch');
    const views = el('div', 'qc-tc-views');
    this.parts = {};
    for (const role of ROLES) {
      const fig = el('figure', 'qc-tc-view'); fig.dataset.view = role;
      const caption = el('figcaption', 'qc-tc-caption'), name = el('span', 'qc-tc-view-name');
      caption.append(name);
      const frame = el('div', 'qc-tc-frame');
      const canvas = el('canvas', 'qc-tc-canvas'); canvas.dataset.role = role; canvas.dataset.side = SIDES[role];
      canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', role === 'control' ? T.asNowLabel : T.testLabel);
      const wait = el('p', 'qc-tc-wait', T.waiting);
      frame.append(canvas, wait);
      const count = el('p', 'qc-tc-count'); count.dataset.suspectsIn = role; count.title = T.suspectsTooltip;
      const total = el('p', 'qc-tc-total'); total.title = T.totalTooltip;
      fig.append(caption, frame, count, total); views.append(fig);
      this.parts[role] = { fig, name, frame, canvas, wait, count, total, px: 0 };
    }
    const progress = this.progressBox = el('div', 'qc-tc-progress');
    const track = el('div', 'qc-tc-track'); track.setAttribute('aria-hidden', 'true');
    this.fills = {};
    for (const role of ROLES) {
      const f = el('i', 'qc-tc-fill'); f.dataset.run = role; f.dataset.side = SIDES[role]; f.style.width = '0%'; track.append(f); this.fills[role] = f;
    }
    this.marks = MARKS.map(at => { const mark = el('i', 'qc-tc-mark'); mark.style.left = `${((at / OF) * 100).toFixed(2)}%`; track.append(mark); return mark; });
    this.progressLine = el('p', 'qc-tc-progress-line');
    this.cancelButton = this.button(T.cancel, 'cancel'); this.cancelButton.title = T.cancelTooltip;
    const progressRow = el('div', 'qc-tc-progress-row'); progressRow.append(this.progressLine, this.cancelButton);
    this.runsLine = el('p', 'qc-tc-runs-line'); this.runsLine.hidden = true;
    this.retryNode = el('p', 'qc-tc-retry'); this.retryNode.hidden = true;
    progress.append(track, progressRow, this.runsLine, this.retryNode);
    const scrub = this.scrubBox = el('div', 'qc-tc-scrub');
    this.replayButton = this.tool('replay', ICONS.play, T.replay); this.replayButton.title = T.replayTooltip; this.replayButton.setAttribute('aria-pressed', 'false');
    this.scrubber = el('input', 'qc-tc-scrubber'); this.scrubber.type = 'range'; this.scrubber.dataset.action = 'scrub';
    this.scrubber.min = '0'; this.scrubber.max = '0'; this.scrubber.step = '1'; this.scrubber.value = '0';
    this.scrubber.setAttribute('aria-label', T.scrubLabel); this.scrubber.title = T.scrubTooltip;
    this.scrubAt = el('span', 'qc-tc-scrub-at');
    scrub.append(this.replayButton, this.scrubber, this.scrubAt);
    this.legend = el('div', 'qc-tc-legend'); this.keys = {};
    for (const [k, text, tip] of [['cell', T.keyCell, T.keyCellTooltip], ['nb', T.keyNeighbours, T.keyNeighboursTooltip],
      ['suspect', T.keySuspect, T.keySuspectTooltip], ['moved', T.keyMoved, T.keyMovedTooltip]]) {
      const item = this.keys[k] = el('span', `qc-tc-key qc-tc-key-${k}`); item.title = tip; item.append(el('i'), el('span', '', text)); this.legend.append(item);
    }
    // Under the views, muted: the suspects the runs couldn't tell apart (a grey ring, as drawn), and the views at the
    // answer that could not be had (what they show may be an earlier iteration's).
    this.unsureNode = this.unsureLine(T.unsureTooltip);
    this.staleNode = el('p', 'qc-tc-stale', T.stale); this.staleNode.title = T.staleTooltip; this.staleNode.hidden = true;
    watch.append(views, this.unsureNode, this.staleNode, progress, scrub, this.legend);
    // The answer, under the views or beside them on a wide stage (its root carries the engine's verdict: data-verdict):
    // after Run again the earlier answer's word and why a repeat is no stronger answer; then its parts (PARTS), each a
    // section of its own with its hooks; New run with this, Keep my setting, Run again (none of them the main one).
    const answer = this.answerBox = el('section', 'qc-tc-answer');
    this.repeatNode = el('div', 'qc-tc-repeat'); this.repeatNode.hidden = true;
    this.repeatText = el('p', 'qc-tc-repeat-text'); this.repeatNote = el('p', 'qc-tc-repeat-note');
    this.repeatNode.append(this.repeatText, this.repeatNote);
    const parts = this.answerParts = {};
    const part = (key, titled = true) => {
      const root = el('section', 'qc-tc-part'); root.dataset.part = key; root.hidden = true;
      const p = parts[key] = { root };
      if (titled) { p.head = el('h3', 'qc-tc-part-head'); p.title = el('span', 'qc-tc-part-title'); p.head.append(p.title); root.append(p.head); }
      return p;
    };
    const back = part('back');
    back.word = el('span', 'qc-tc-word'); back.head.append(back.word);
    // Where the test was not measured, when no "fewer" part carries it in its ⓘ (they came and went).
    back.info = this.info(); back.info.setAttribute('aria-label', T.caveatsLabel); back.info.hidden = true; back.head.append(back.info);
    back.text = el('p', 'qc-tc-text'); back.root.append(back.text);
    const fewer = part('fewer');
    fewer.info = this.info(); fewer.info.setAttribute('aria-label', T.aboutLabel); fewer.head.append(fewer.info);
    fewer.verdict = el('p', 'qc-tc-verdict'); fewer.text = el('p', 'qc-tc-text'); fewer.why = el('p', 'qc-tc-why');
    fewer.root.append(fewer.verdict, fewer.text, fewer.why);
    const cost = part('cost'), rows = el('div', 'qc-tc-rows');
    cost.rows = {};
    for (const key of COST_ROWS) {
      const row = el('div', 'qc-tc-row'); row.dataset.row = key;
      const r = cost.rows[key] = { root: row, label: el('span', 'qc-tc-row-label'), value: el('span', 'qc-tc-row-value'), change: el('span', 'qc-tc-row-change') };
      row.append(r.label, r.value, r.change); rows.append(row);
    }
    cost.shrink = el('p', 'qc-tc-shrink'); cost.shrink.hidden = true;
    cost.root.append(rows, cost.shrink);
    const went = part('went');
    went.bars = {};
    for (const side of ['A', 'B']) {
      const line = el('div', 'qc-tc-went'); line.dataset.side = side;
      const label = el('span', 'qc-tc-went-side'), bar = el('div', 'qc-tc-bar'); bar.setAttribute('role', 'img');
      const segs = {};
      for (const key of WENT) { const seg = el('i'); seg.dataset.went = key; seg.style.width = '0%'; bar.append(seg); segs[key] = seg; }
      line.append(label, bar); went.root.append(line);
      went.bars[side] = { root: line, label, bar, segs };
    }
    went.key = el('p', 'qc-tc-went-key'); went.root.append(went.key); this.wentKey = null;
    const alone = part('alone', false), toofew = part('toofew', false);
    alone.text = el('p', 'qc-tc-text'); alone.root.append(alone.text);
    toofew.text = el('p', 'qc-tc-text'); toofew.root.append(toofew.text);
    this.newButton = this.button('', 'new'); this.keepButton = this.button('', 'keep'); this.againButton = this.button(T.again, 'again');
    const answerActions = el('div', 'qc-tc-actions'); answerActions.append(this.newButton, this.keepButton, this.againButton);
    answer.append(this.repeatNode, ...PARTS.map(key => parts[key].root), answerActions);
    // A test that stopped: what happened, Close.
    const ended = this.endedBox = el('section', 'qc-tc-ended');
    this.message = el('p', 'qc-tc-message');
    this.endedClose = this.button(T.close, 'close'); this.endedClose.title = T.closeTooltip;
    const endedActions = el('div', 'qc-tc-actions'); endedActions.append(this.endedClose);
    ended.append(this.message, endedActions);
    // What a press could not do (the panel's words), in the section shown; a stopped test says it in its own section.
    this.alert = el('p', 'qc-tc-alert'); this.alert.setAttribute('role', 'alert'); this.alert.hidden = true;
    body.append(form, waiting, watch, answer, ended);
    root.append(head, body);
    this.host?.append?.(root);
    // ---- behaviour
    this.smallButton.onclick = event => { event?.stopPropagation?.(); this.setSmall(!this.minimised); };
    this.closeButton.onclick = event => { event?.stopPropagation?.(); this.act('close'); };
    this.replayButton.onclick = event => { event?.stopPropagation?.(); if (this.replaying) this.stopReplay(); else this.replay(); };
    this.select.onchange = () => this.act('setting', this.select.value);
    this.input.oninput = () => this.act('value', this.input.value);
    this.choiceBox.onclick = event => {
      let b = event?.target ?? null;
      while (b && !(b.tagName === 'BUTTON' && b.dataset?.value != null)) b = b.parentElement;
      if (b && !b.disabled) this.act('value', b.dataset.value);
    };
    this.listen(this.scrubber, 'input', () => { this.stopReplay(); this.scrubTo(Number(this.scrubber.value)); });
    this.listen(root, 'keydown', e => {
      if (e.key !== 'Escape' || !CLOSABLE.has(this.phase)) return;
      e.preventDefault?.(); e.stopPropagation?.(); this.act('close');
    });
    // Drag the header to move; double-click docks it again at the bottom left.
    this.listen(head, 'pointerdown', e => {
      if (e.button !== 0 || e.target?.closest?.('button')) return;
      e.preventDefault?.(); try { head.setPointerCapture?.(e.pointerId); } catch { /* not captured */ }
      this.drag = { id: e.pointerId, x: e.clientX, y: e.clientY, left: this.left, bottom: this.bottom, moved: false };
    });
    this.listen(head, 'pointermove', e => {
      const d = this.drag; if (!d || d.id !== e.pointerId) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) < 3) return;
      d.moved = true; this.moveTo(d.left + dx, d.bottom - dy);
    });
    const release = e => { const d = this.drag; if (!d || (e && d.id !== e.pointerId)) return; this.drag = null; };
    this.listen(head, 'pointerup', release); this.listen(head, 'pointercancel', release); this.listen(head, 'lostpointercapture', () => release(null));
    this.listen(head, 'dblclick', e => { if (e.target?.closest?.('button')) return; this.moveTo(MARGIN, MARGIN); });
    // The corner's size, as the browser reports it (the Mixing legend moves out of its way; it stays inside the stage,
    // also when it grows: the answer's column beside the views).
    if (typeof globalThis.ResizeObserver === 'function') {
      this.observer = new globalThis.ResizeObserver(entries => {
        const box = entries.at(-1)?.borderBoxSize?.[0], rect = entries.at(-1)?.contentRect;
        const w = box?.inlineSize ?? rect?.width ?? 0, h = box?.blockSize ?? rect?.height ?? 0;
        if (w > 0 && h > 0 && (!this.size || Math.abs(w - this.size.w) > 0.5 || Math.abs(h - this.size.h) > 0.5)) {
          this.size = { w, h }; if (this.isOpen) this.moveTo(this.left, this.bottom);
        }
      });
      this.observer.observe(root);
    }
  }

  // ---- what the panel says ------------------------------------------------------------------------------------------
  /** The panel's test as it stands (qc-panel.js testModel: {open, phase, run, cell, start, form {settings, flag, value,
   *  problems, notHere, field, pool, area, runs, reasons}, test {…, valueText, runsText, cancelTooltip}, progress, result,
   *  message, busy, busyText}); open false closes the corner. Cheap when nothing changed (texts are written only when
   *  they differ). */
  show(model) {
    if (!model || typeof model !== 'object' || model.open === false) { this.close(); return; }
    const phase = PHASES.includes(model.phase) ? model.phase : 'form';
    const was = this.isOpen ? this.phase : null;
    if (!this.isOpen) this.open();
    const run = model.run ?? null;
    if (run !== this.runId) { this.runId = run; this.clearTest(); }
    else if (NEW_TEST.has(phase) && !NEW_TEST.has(was)) this.clearTest();   // Run the test, Run again: another test
    if (phase === 'done' && was !== 'done') {                                 // one more ask: the views at the answer
      this.finalWanted = true; this.finalFails = 0; this.retryAt = -Infinity; this.stale = false;
    }
    if (phase !== 'done') this.stopReplay();
    this.model = model; this.phase = phase; this.settle();
    if (this.root.dataset.phase !== phase) this.root.dataset.phase = phase;
    this.render();
    this.layout();   // after the words: the muted lines that show take their room
    this.ask(false); this.wake();
    if (phase === 'form' && was !== 'form') try { this.input.focus?.({ preventScroll: true }); } catch { /* focus is a convenience */ }
  }
  /** A press for the panel. A button pressed again before the panel answered (or while it says that press is on its way,
   *  model.busy) is not sent twice; the form's own choices (FORM_ACTIONS) go as they come. */
  act(action, payload) {
    if (!ACTIONS.includes(action) || !this.isOpen) return;
    if (!FORM_ACTIONS.has(action)) {
      if (this.pending === action || this.model?.busy === action) return;
      this.pending = action; this.later('pending', PENDING_MS, () => this.settle());
      for (const b of this.buttons()) if (b.dataset.action === action && b.getAttribute('aria-busy') !== 'true') b.setAttribute('aria-busy', 'true');
    }
    this.call('onAction', action, payload);
  }
  /** The panel answered (or did not in time): the buttons can be pressed again. */
  settle() {
    this.pending = null; this.cancel('pending');
    const busy = this.model?.busy;
    for (const b of this.buttons()) if (b.getAttribute('aria-busy') === 'true' && b.dataset.action !== busy) b.removeAttribute('aria-busy');
  }
  buttons() {
    return [this.runButton, this.waitCancel, this.cancelButton, this.newButton, this.keepButton, this.againButton, this.endedClose, this.closeButton];
  }

  open() {
    this.isOpen = true; this.root.hidden = false;
    this.dpr = clamp(globalThis.devicePixelRatio || 1, 1, 3);
    if (this.minimised) this.setSmall(false, false);
  }
  /** Closed: nothing scheduled or asked any more, the frames forgotten. */
  close() {
    if (!this.isOpen) return;
    this.demo?.close();
    this.isOpen = false; this.root.hidden = true; this.root.dataset.phase = 'closed';
    this.phase = null; this.model = null; this.runId = null; this.drag = null;
    this.settle();
    for (const key of [...this.timers.keys()]) this.cancel(key);
    if (this.paintFrame != null) { this.clock.cancelAnimationFrame(this.paintFrame); this.paintFrame = null; }
    this.clearTest(); this.glowTimes = [];
    if (this.minimised) this.setSmall(false, false);
    this.setBeside(false); this.shownLines = { views: false, stale: false, retry: false };
    this.size = null;   // the browser reports its size again when it opens
    this.boxKey = null; this.call('onBox', null);
  }
  /** Another test (or none): its frames, the scrubber and any ask on its way belong to the old one. */
  clearTest() {
    this.gen++;
    this.frames = { control: [], test: [] }; this.steps = []; this.shown = { control: null, test: null }; this.fades = { control: null, test: null };
    this.dirty = { control: true, test: true }; this.scrub = null; this.crop = null; this.cropKey = ''; this.suspects = []; this.suspectsKey = null;
    this.composites = []; this.viewEnded = false; this.finalWanted = false; this.fetchedAt = -Infinity;
    this.finalFails = 0; this.retryAt = -Infinity; this.stale = false;
    this.stopReplay();
  }
  /** Make smaller (the header only, its line saying how far the test is) or bigger, where it is. */
  setSmall(on, render = true) {
    this.minimised = !!on;
    this.root.classList?.toggle('qc-tc-small', this.minimised);
    this.smallButton.innerHTML = this.minimised ? ICONS.restore : ICONS.minimise;
    const label = this.minimised ? T.restore : T.minimise;
    this.smallButton.title = label; this.smallButton.setAttribute('aria-label', label); this.smallButton.setAttribute('aria-pressed', String(this.minimised));
    if (this.minimised) this.stopPaint(); else this.dirty = { control: true, test: true };
    if (!render || !this.isOpen) return;
    this.render(); this.layout(); this.wake();   // the strip is never the wide card
  }
  visibility() {
    if (!this.isOpen) return;
    if (this.hidden) { this.stopPaint(); this.cancel('ask'); this.stopReplay(); return; }
    this.dirty = { control: true, test: true }; this.wake(); this.ask(false);
  }
  /** A full-screen view covers the page (main.js: Compare's dialog), or no longer does: covered, the corner asks and
   *  paints nothing, as while the page is hidden; uncovered, it catches up (visibility). */
  setCovered(on) {
    on = !!on;
    if (on === this.covered) return;
    this.covered = on;
    this.visibility();
  }
  /** The motion setting changed (calm motion, reduced motion): the glow breathes again or holds still. */
  refresh() { if (!this.isOpen) return; this.dirty = { control: true, test: true }; this.wake(); }
  /** The page's colours changed (Colour by, new clusters: colorOf answers otherwise): the views are drawn again in them. */
  recolor() { if (!this.isOpen) return; this.invalidate(); this.renderLegend(); this.wake(); }
  stopPaint() {
    this.cancel('paint');
    if (this.paintFrame != null) { this.clock.cancelAnimationFrame(this.paintFrame); this.paintFrame = null; }
  }

  // ---- asking for the view ------------------------------------------------------------------------------------------
  wantsView() {
    if (!this.isOpen || this.hidden || this.runId == null || typeof this.o.fetchView !== 'function') return false;
    if (this.phase === 'done') return this.finalWanted;
    return ASKING.has(this.phase) && !this.viewEnded;
  }
  /** Ask for the view when it is wanted: never sooner than VIEW_MS after the last ask (nor before a retry is due), one
   *  at a time. `now`: the timer fired (else only scheduled). */
  ask(now) {
    if (!this.wantsView() || this.fetching) return;
    const wait = Math.max(this.fetchedAt + VIEW_MS, this.retryAt) - this.now();
    if (!now || wait > 0) { if (!this.timers.has('ask')) this.later('ask', wait, () => this.ask(true)); return; }
    this.fetchNow();
  }
  async fetchNow() {
    const gen = this.gen, run = this.runId, final = this.phase === 'done';
    this.fetching = true; this.fetchedAt = this.now(); this.fetches++; this.hook?.count('qcTestViewFetches');
    let reply = null;
    try { reply = await this.o.fetchView(run); } catch { reply = null; }   // quiet: the path is under /qc
    this.fetching = false;
    if (gen === this.gen && this.isOpen) {
      const took = !!reply && this.take(reply, final);
      if (final && this.phase === 'done') this.answered(took);
    }
    this.ask(false);
  }
  /** The ask for the views at the answer came back: `took` the engine's last view (they are no longer wanted), else it
   *  failed or was not that, and is made again RETRY_MS after, then twice as long each time, FINAL_TRIES asks in all;
   *  then nothing more is asked and the views say they may be an earlier iteration's. */
  answered(took) {
    if (took) { this.finalWanted = false; this.finalFails = 0; this.retryAt = -Infinity; return; }
    this.finalFails++;
    if (this.finalFails < FINAL_TRIES) { this.retryAt = this.now() + RETRY_MS * 2 ** (this.finalFails - 1); return; }
    this.finalWanted = false; this.stale = true;
    if (VIEWING.has(this.phase)) this.renderViews();
  }
  /** A view reply: the crop, the suspects, and each run's newest frame kept if it is new. `final` (the ask at the
   *  answer): taken only when it says the test ended (the engine's last views). Returns whether it was taken. */
  take(raw, final = false) {
    const v = normaliseView(raw);
    if (!v || (final && !VIEW_ENDED.has(v.state))) return false;
    const sKey = v.suspects.map(s => `${s.x},${s.y},${s.gene},${s.p}`).join(';');
    if (sKey !== this.suspectsKey) {   // they come with the test: other ones mean the frames kept were for another list
      if (this.suspectsKey != null) {
        this.frames = { control: [], test: [] }; this.steps = []; this.scrub = null; this.stopReplay();
        this.shown = { control: null, test: null }; this.fades = { control: null, test: null };
      }
      this.suspects = v.suspects; this.suspectsKey = sKey; this.invalidate();
    }
    if (v.box && v.box.join(',') !== this.cropKey) { this.crop = v.box; this.cropKey = v.box.join(','); this.invalidate(); }
    let added = false;
    for (const run of v.runs) added = this.keep(run) || added;
    if (VIEW_ENDED.has(v.state)) this.viewEnded = true;
    if (added) {
      const step = { control: this.frames.control.at(-1)?.seq ?? null, test: this.frames.test.at(-1)?.seq ?? null }, last = this.steps.at(-1);
      if (!last || last.control !== step.control || last.test !== step.test) this.steps.push(step);
    }
    if (this.scrub == null) for (const role of ROLES) this.setShown(role, this.frames[role].at(-1) ?? null);
    if (this.isOpen && VIEWING.has(this.phase)) this.renderViews();
    this.wake();
    return true;
  }
  /** One run's frame, kept unless it is already (a frame is decoded once). A frame number lower than the newest kept means
   *  the run started again (a retry): the frames from there on are dropped. */
  keep(run) {
    let kept = this.frames[run.role];
    if (kept.length && run.seq < kept.at(-1).seq) {
      kept = this.frames[run.role] = kept.filter(f => f.seq < run.seq);
      this.steps = this.steps.filter(s => s[run.role] == null || kept.some(f => f.seq === s[run.role]));
      if (this.scrub != null) this.scrub = Math.min(this.scrub, Math.max(0, this.steps.length - 1));
    }
    if (kept.some(f => f.seq === run.seq)) return false;
    const labels = decodeLabels(run.labels, run.w * run.h);
    if (!labels) return false;
    const counted = shares(run.held, run.cell);   // weights, moved, kept, of
    const frame = { role: run.role, seq: run.seq, iter: run.iter, voxel: run.voxel, x0: run.x0, y0: run.y0, w: run.w, h: run.h,
      cell: run.cell, sources: run.sources, parents: run.parents, shape: outline(labels, run.w, run.h, run.cell, run.sources),
      held: run.held, heldBy: run.heldBy, ...counted, lit: counted.weights.reduce((a, v, k) => a + (v > 0 && this.suspects[k]?.ok ? 1 : 0), 0),
      unsure: run.unsure, unsureCount: run.unsure.reduce((a, v) => a + v, 0), cellTotal: run.cellTotal, comp: null, glow: null };
    let i = kept.length; while (i > 0 && kept[i - 1].seq > frame.seq) i--;
    kept.splice(i, 0, frame);
    if (kept.length > MAX_FRAMES) kept.splice(1, 1);   // the first one stays: the replay starts where the test did
    return true;
  }
  /** Drawn frames no longer fit (another crop, size, set of suspects or the page's colours): all are drawn again. */
  invalidate() {
    for (const f of [...this.frames.control, ...this.frames.test]) { f.comp = null; f.glow = null; }
    this.composites = []; this.geom = null; this.dirty = { control: true, test: true };
  }
  /** `frame` is what a view shows now: it fades in over the one shown before (not under reduced or calm motion). */
  setShown(role, frame) {
    const before = this.shown[role];
    if (before === frame) return;
    this.shown[role] = frame;
    this.fades[role] = before && frame && !this.still ? { from: before, start: this.now() } : null;
    this.dirty[role] = true;
  }

  // ---- the scrubber ---------------------------------------------------------------------------------------------------
  /** Step `k` of the frames kept (what both views showed after each ask); the last step follows the newest again. */
  scrubTo(k) {
    if (!this.steps.length) return;
    k = clamp(Math.round(finite(k) ? k : 0), 0, this.steps.length - 1);
    this.scrub = k >= this.steps.length - 1 ? null : k;
    const step = this.steps[k];
    for (const role of ROLES) {
      const seq = this.scrub == null ? null : step[role];
      this.setShown(role, this.scrub == null ? this.frames[role].at(-1) ?? null : seq == null ? null : this.frames[role].find(f => f.seq === seq) ?? null);
    }
    this.renderViews(); this.wake();
  }
  /** Replay: from the first kept step to the last, one step every REPLAY_MS. */
  replay() {
    if (this.steps.length < 2 || !this.isOpen || this.hidden) return;
    this.replaying = true; this.replayButton.innerHTML = ICONS.stop; this.replayButton.setAttribute('aria-pressed', 'true');
    this.replayButton.setAttribute('aria-label', T.replayStop);
    this.scrubTo(0);
    const next = () => {
      const k = (this.scrub ?? this.steps.length - 1) + 1;
      this.scrubTo(k);
      if (this.scrub == null) this.stopReplay(); else this.later('replay', REPLAY_MS, next);
    };
    this.later('replay', REPLAY_MS, next);
  }
  stopReplay() {
    this.cancel('replay');
    if (!this.replaying) return;
    this.replaying = false;
    if (this.replayButton) { this.replayButton.innerHTML = ICONS.play; this.replayButton.setAttribute('aria-pressed', 'false'); this.replayButton.setAttribute('aria-label', T.replay); }
  }

  // ---- words ----------------------------------------------------------------------------------------------------------
  render() {
    const m = this.model, phase = this.phase;
    if (!m) return;
    const viewing = VIEWING.has(phase), ended = ENDED.has(phase), form = m.form && typeof m.form === 'object' ? m.form : {};
    setText(this.titleNode, phase === 'form' ? T.titleForm : phase === 'done' ? T.titleDone : ended ? T.titleEnded : T.titleRunning);
    const sub = this.minimised && NEW_TEST.has(phase) ? this.progressText() : this.subText();
    setText(this.subNode, sub); setTitle(this.subNode, sub);   // cut short on a narrow corner: the whole of it on hover
    setHidden(this.closeButton, !CLOSABLE.has(phase));
    setHidden(this.body, this.minimised);
    setHidden(this.formBox, phase !== 'form'); setHidden(this.waitBox, phase !== 'waiting'); setHidden(this.watchBox, !viewing);
    setHidden(this.answerBox, phase !== 'done'); setHidden(this.endedBox, !ended);
    if (phase === 'form') this.renderForm(form);
    if (phase === 'waiting') this.renderWaiting(m.waiting); else setData(this.waitingLine, 'waiting', null);   // its hook only while it waits
    if (viewing) this.renderViews();
    else { this.setLine('views', false); this.setLine('stale', false); this.setLine('retry', false); }
    this.renderAnswer(phase === 'done' && m.result && typeof m.result === 'object' ? m.result : null);
    if (ended) setText(this.message, plain(m.message) || (phase === 'failed' ? T.failed : T.cancelled));
    this.renderButtons();
    const where = phase === 'done' ? this.answerBox : viewing ? this.watchBox : phase === 'waiting' ? this.waitBox : this.formBox;
    const alert = ended ? '' : plain(m.message);
    if (this.alert.parentElement !== where) where.append(this.alert);
    setText(this.alert, alert); setHidden(this.alert, !alert);
  }
  /** The buttons' words (the panel's for the answers and Cancel's tooltip, the corner's own else) and which one is
   *  working: the one the panel says is on its way (model.busy) reads its words for that ("Cancelling…") and carries
   *  aria-busy, never disabled. An answer the panel gives no words for is not offered; none of the answers is the main
   *  one (nothing is suggested). Run's tooltip says, after what it does, what a press would wait for first and where the
   *  test was not measured (the form's waits and caveats, a line each). */
  renderButtons() {
    const m = this.model ?? {}, r = m.result && typeof m.result === 'object' ? m.result : {}, t = m.test && typeof m.test === 'object' ? m.test : {};
    const form = m.form && typeof m.form === 'object' ? m.form : {};
    const busy = typeof m.busy === 'string' ? m.busy : null, doing = plain(m.busyText);
    const words = { run: [T.run, [T.runTooltip, ...textsOf(form.waits), ...textsOf(form.caveats)].join('\n')],
      cancel: [T.cancel, plain(t.cancelTooltip) || T.cancelTooltip], close: [T.close, T.closeTooltip],
      again: [plain(r.again?.text) || T.again, plain(r.again?.tooltip) || T.againTooltip],
      new: [plain(r.new?.text), plain(r.new?.tooltip)], keep: [plain(r.keep?.text), plain(r.keep?.tooltip)] };
    for (const b of [this.runButton, this.waitCancel, this.cancelButton, this.newButton, this.keepButton, this.againButton, this.endedClose]) {
      const action = b.dataset.action, [text, tip] = words[action], working = busy === action || this.pending === action;
      setText(b, busy === action && doing ? doing : text); setTitle(b, tip);
      if (action === 'new' || action === 'keep') setHidden(b, !text);
      if (working !== (b.getAttribute('aria-busy') === 'true')) { if (working) b.setAttribute('aria-busy', 'true'); else b.removeAttribute('aria-busy'); }
    }
  }
  /** The test view's title: "With 0.02" (the header names the setting). */
  testName() { return corner('withValue', { v: plain(this.model?.test?.valueText) || T.missing }); }
  /** The header's second line: the setting tested, or chosen in the form, and the cell (data: the names are the
   *  sample's and the panel's). */
  subText() {
    const m = this.model ?? {}, form = m.form ?? {};
    const setting = this.phase === 'form' ? plain(listOf(form.settings).find(s => s.flag === form.flag)?.name) : plain(m.test?.name);
    const cell = plain(m.cell?.name);
    return setting && cell ? corner('sub', { setting, cell }) : setting || cell;
  }
  renderForm(form) {
    const settings = listOf(form.settings);
    this.choices(settings);
    if (form.flag != null && this.select.value !== form.flag) this.select.value = form.flag;
    this.demo?.setFlag(form.flag ?? null);
    // Its value (never moved under the caret), "now 1" (the panel's words; in its tooltip what a test of the hold set
    // live since the run's last iteration compares with: the panel's nowTip), its note with the panel's tooltip for it.
    const chosen = settings.find(s => s.flag === form.flag) ?? null, text = form.value == null ? '' : String(form.value);
    if (this.input.value !== text && this.doc.activeElement !== this.input) this.input.value = text;
    this.renderChoice(chosen, text);
    const now = plain(chosen?.nowText) || (finite(chosen?.now) ? corner('now', { v: settingValue(chosen.now) }) : '');
    setText(this.nowNode, now); setHidden(this.nowNode, !now); setTitle(this.nowNode, now ? plain(chosen?.nowTip) : '');
    setText(this.noteNode, plain(chosen?.note)); setHidden(this.noteNode, !plain(chosen?.note)); setTitle(this.noteNode, plain(chosen?.tip));
    // The panel's lines: the pool (its state and, once known, its numbers as data: the harness reads them), the area
    // (its rule as data), the runs (how they are made in its tooltip: the panel's How line).
    const pool = form.pool && typeof form.pool === 'object' ? form.pool : null, area = form.area && typeof form.area === 'object' ? form.area : null;
    setText(this.poolNode, plain(pool?.text)); setHidden(this.poolNode, !plain(pool?.text)); setTitle(this.poolNode, plain(pool?.tooltip));
    setData(this.poolNode, 'pool', plain(pool?.state) || null);
    for (const key of ['cells', 'suspects', 'need']) setData(this.poolNode, key, Number.isInteger(pool?.[key]) ? String(pool[key]) : null);
    setText(this.areaNode, plain(area?.text)); setHidden(this.areaNode, !plain(area?.text)); setData(this.areaNode, 'area', plain(area?.rule) || null);
    setText(this.runsNode, plain(form.runs)); setHidden(this.runsNode, !plain(form.runs)); setTitle(this.runsNode, plain(form.how));
    // Why a test can't run here: the engine's refusals in the panel's words, each by its code, in their groups (what
    // passes by itself first, what leaves nothing to test or can't run: qc-copy.mjs testReasonGroup), each group under
    // its heading with what it means in the heading's tooltip and only with a reason in it, the engine's order kept inside
    // each, each with the panel's tooltip for it when it has one (`tip`: the count the test reads, at a saved end). What
    // a press would wait for first and where the test was not measured: Run's tooltip (renderButtons), their codes as
    // data.
    setData(this.runButton, 'waits', codesOf(form.waits)); setData(this.runButton, 'caveats', codesOf(form.caveats));
    const reasons = (Array.isArray(form.reasons) ? form.reasons : []).filter(r => r && typeof r.code === 'string' && plain(r.text));
    const rKey = reasons.map(r => `${r.code}\u0000${plain(r.text)}\u0000${plain(r.tip)}`).join('\u0001');
    if (rKey !== this.reasonsKey) {
      this.reasonsKey = rKey;
      this.reasons.replaceChildren(...TEST_REASON_GROUPS.flatMap(group => {
        const listed = reasons.filter(r => testReasonGroup(r.code) === group);
        if (!listed.length) return [];
        const box = this.el('div', 'qc-tc-reason-group'), [text, tip] = REASON_HEADS[group], head = this.el('p', 'qc-tc-reason-head', text);
        box.dataset.group = group; head.title = tip;
        const list = this.el('ul');
        for (const r of listed) { const li = this.el('li', '', plain(r.text)); li.dataset.reason = r.code; if (plain(r.tip)) li.title = plain(r.tip); list.append(li); }
        box.append(head, list);
        return [box];
      }));
    }
    setHidden(this.reasons, !reasons.length);
    // The page's own problems under "Check these first:"; a press while the engine's refusals are listed says the panel's
    // one line instead (`notHere`), never as something to check.
    const problems = (Array.isArray(form.problems) ? form.problems : []).map(plain).filter(Boolean), notHere = plain(form.notHere);
    const pKey = `${problems.join('\u0001')}\u0002${notHere}`;
    if (pKey !== this.problemsKey) {
      this.problemsKey = pKey;
      const list = this.el('ul'); for (const line of problems) list.append(this.el('li', '', line));
      this.problems.replaceChildren(...(problems.length ? [this.el('p', '', T.problemsLead), list] : []), ...(notHere ? [this.el('p', 'qc-tc-not-here', notHere)] : []));
    }
    setHidden(this.problems, !problems.length && !notHere);
    // The panel names the field the problems are about (`field`: 'value').
    const marked = problems.length > 0 && form.field === 'value';
    if (marked !== this.input.classList.contains('needs-input')) {
      this.input.classList.toggle('needs-input', marked);
      if (marked) this.input.setAttribute('aria-invalid', 'true'); else this.input.removeAttribute('aria-invalid');
    }
  }
  /** The value's place by the chosen setting's kind (the panel's `kind`): a typed number (whole numbers step by 1), or
   *  its values as buttons (`choices` [{value, text, now}]: the run's own one disabled, the one picked pressed). */
  renderChoice(chosen, text) {
    const kind = chosen?.kind ?? 'number', picked = kind === 'choice' || kind === 'switch';
    const step = kind === 'whole' ? '1' : 'any', mode = kind === 'whole' ? 'numeric' : 'decimal';
    if (this.input.step !== step) this.input.step = step;
    if (this.input.getAttribute?.('inputmode') !== mode) this.input.setAttribute('inputmode', mode);
    setHidden(this.input, picked); setHidden(this.choiceBox, !picked);
    const choices = picked && Array.isArray(chosen?.choices) ? chosen.choices : [];
    const key = picked ? `${chosen.flag}\u0001${choices.map(c => `${c.value}\u0000${c.text}\u0000${!!c.now}`).join('\u0001')}` : null;
    if (key !== this.choiceKey) {
      this.choiceKey = key;
      this.choiceBox.replaceChildren(...choices.map(c => {
        const b = this.el('button', 'qc-tc-pick', plain(c.text)); b.type = 'button'; b.dataset.value = String(c.value);
        b.setAttribute('role', 'radio'); b.disabled = !!c.now; if (c.now) b.title = T.choiceNowTooltip;
        return b;
      }));
    }
    for (const b of this.choiceBox.children ?? []) {
      const on = String(b.dataset.value) === text;
      if (b.getAttribute?.('aria-checked') !== String(on)) b.setAttribute('aria-checked', String(on));
    }
    this.choiceBox.dataset.kind = picked ? kind : '';
  }
  /** A test that waits: what for, in the panel's words (the first; every one in the line's tooltip when there are more;
   *  once none holds, what the engine does before its runs start; "Starting…" without words), their codes as data. */
  renderWaiting(waiting) {
    const w = waiting && typeof waiting === 'object' ? waiting : {};
    setText(this.waitingLine, plain(w.text) || T.starting); setTitle(this.waitingLine, plain(w.tooltip));
    setData(this.waitingLine, 'waiting', codesOf(w.codes));
  }
  /** The setting's choices (the panel's, each saying whether a test tried it), written again only when they change. */
  choices(settings) {
    const key = settings.map(s => `${s.flag}\u0000${s.name}\u0000${s.tried !== false}`).join('\u0001');
    if (key === this.settingsKey) return;
    this.settingsKey = key;
    this.select.replaceChildren(...settings.map(s => {
      const o = this.el('option', '', plain(s.name) || s.flag); o.value = s.flag; o.setAttribute('value', s.flag); o.dataset.tried = String(s.tried !== false);
      return o;
    }));
  }
  /** The views' words: their titles, the numbers under them, the suspects the runs couldn't tell apart, the progress
   *  (the engine's: a bar per side, the schedule's parts marked, the panel's lines) or the scrubber, the key. */
  renderViews() {
    const m = this.model ?? {}, phase = this.phase, name = this.testName();
    setText(this.parts.control.name, T.asNow); setText(this.parts.test.name, name); setTitle(this.parts.test.name, name);
    const sent = this.suspects.length;   // the view's own suspects (the pool's of the cell), once one came
    const any = this.frames.control.length + this.frames.test.length > 0;
    for (const role of ROLES) {
      const p = this.parts[role], f = this.shown[role];
      // "of n": the suspects with a share in the run on screen (before one, every suspect sent).
      const n = f ? f.of : Math.max(0, sent);
      setText(p.wait, phase === 'done' && !any ? T.noFrames : T.waiting); setHidden(p.wait, !!f);
      if (f && f.cell == null) {
        setText(p.count, T.notFound); setTitle(p.count, T.notFoundTooltip); setHidden(p.count, false);
        setData(p.count, 'count', ''); setData(p.count, 'of', String(n)); setData(p.count, 'held', '');
      } else if (f && !n && sent > 0) {             // the cell is there, none of its suspects placed: said, not hidden
        setText(p.count, T.notPlaced); setTitle(p.count, T.notPlacedTooltip); setHidden(p.count, false);
        setData(p.count, 'count', ''); setData(p.count, 'of', '0'); setData(p.count, 'held', f.heldBy ?? '');
      } else {
        // Approximate shares: a later iteration may bring proseg's exact ones while the test runs, none once its runs
        // ended (its answer being worked out, or the test over).
        const approx = RUNS_ENDED.has(phase) ? T.approxDoneTooltip : T.approxTooltip;
        const tip = f?.heldBy === 'approx' ? `${T.suspectsTooltip} ${approx}` : T.suspectsTooltip;
        setText(p.count, n ? corner('suspectsInside', { a: f ? number(f.kept) : T.missing, n: number(n) }) : ''); setTitle(p.count, tip);
        setHidden(p.count, !n); setData(p.count, 'count', f ? String(f.kept) : ''); setData(p.count, 'of', String(n));
        setData(p.count, 'held', f?.heldBy ?? '');
      }
      const total = f && f.cell != null && f.cellTotal != null;
      setText(p.total, total ? corner('transcripts', { t: number(f.cellTotal) }) : ''); setHidden(p.total, !total);
    }
    setHidden(this.legend, !any);
    if (any) this.renderLegend();
    setHidden(this.keys.suspect, !sent); setHidden(this.keys.moved, !sent);   // a view without suspects draws none
    // The suspects either view couldn't tell apart, in one muted line.
    const unsure = this.unsureNow();
    this.setUnsure(this.unsureNode, unsure); this.setLine('views', unsure > 0);
    const stale = phase === 'done' && this.stale && any;   // with no frame the view says none was kept
    setHidden(this.staleNode, !stale); this.setLine('stale', stale);
    const going = NEW_TEST.has(phase), progress = m.progress && typeof m.progress === 'object' ? m.progress : null;
    setHidden(this.progressBox, !going); setHidden(this.cancelButton, phase === 'starting');   // Cancel once the test exists
    if (going) {
      setText(this.progressLine, this.progressText()); setTitle(this.progressLine, plain(progress?.tooltip));
      const of = this.of();
      for (const role of ROLES) {
        const side = progress?.sides?.[SIDES[role]], it = phase === 'comparing' ? of : finite(side) ? side : this.shown[role]?.iter;
        const width = `${(finite(it) ? clamp(it / of, 0, 1) * 100 : 0).toFixed(1)}%`;
        if (this.fills[role].style.width !== width) this.fills[role].style.width = width;
      }
      this.marksOf().forEach((at, k) => {
        const left = `${(clamp(at / of, 0, 1) * 100).toFixed(2)}%`;
        if (this.marks[k] && this.marks[k].style.left !== left) this.marks[k].style.left = left;
      });
      const runs = plain(m.test?.runsText), retry = plain(progress?.retry);
      setText(this.runsLine, runs); setHidden(this.runsLine, !runs); setTitle(this.runsLine, plain(m.test?.how));
      setText(this.retryNode, retry); setHidden(this.retryNode, !retry); this.setLine('retry', !!retry);
    } else this.setLine('retry', false);
    const scrubbing = phase === 'done' && this.steps.length > 1;
    setHidden(this.scrubBox, !scrubbing);
    if (scrubbing) {
      const max = String(this.steps.length - 1), at = String(this.scrub ?? this.steps.length - 1);
      if (this.scrubber.max !== max) this.scrubber.max = max;
      if (this.scrubber.value !== at) this.scrubber.value = at;
      const a = this.shown.control?.iter, b = this.shown.test?.iter;
      setText(this.scrubAt, corner('scrubAt', { a: finite(a) ? number(a) : T.missing, b: finite(b) ? number(b) : T.missing }));
    }
    if (this.minimised && going) { const line = this.progressText(); setText(this.subNode, line); setTitle(this.subNode, line); }
  }
  /** How many suspects the views on screen couldn't tell apart (in either: the runs hold the sample's same transcripts,
   *  so it is one number in practice), 0 before a view. */
  unsureNow() {
    const a = this.shown.control?.unsure, b = this.shown.test?.unsure;
    let n = 0;
    if (a || b) for (let k = 0; k < this.suspects.length; k++) if (a?.[k] || b?.[k]) n++;
    return n;
  }
  /** A muted line of the suspects the runs couldn't tell apart: "2 couldn't be told apart …", hidden for none. */
  setUnsure(node, count) {
    setText(node.children[1], count > 0 ? corner('unsure', { n: number(count) }) : ''); setHidden(node, !(count > 0));
  }
  /** How the other cells are coloured: as on the image (the view names their cells in your run and the page colours
   *  them), else each in a colour of its own. Said in the cell's key. */
  colouring() {
    const f = this.shown.control ?? this.shown.test ?? this.frames.control.at(-1) ?? this.frames.test.at(-1);
    return f ? (f.parents && typeof this.o.colorOf === 'function' ? 'stage' : 'own') : null;
  }
  renderLegend() {
    const how = this.colouring();
    setTitle(this.keys.cell, how ? `${T.keyCellTooltip} ${how === 'stage' ? T.colorsStage : T.colorsOwn}` : T.keyCellTooltip);
  }
  /** The schedule's end and its parts' marks (the engine's progress, else proseg's default schedule). */
  of() { const of = this.model?.progress?.of; return finite(of) && of > 0 ? of : OF; }
  marksOf() { const marks = this.model?.progress?.marks; return Array.isArray(marks) && marks.length === 2 && marks.every(finite) ? marks : MARKS; }
  /** What the test is doing, in the panel's words ("Iteration 212 of 450 · refining", "Reading the runs…"; what it waits
   *  for first), or "Starting…" before the test exists. */
  progressText() {
    if (this.phase === 'waiting') return plain(this.model?.waiting?.text) || T.starting;
    return (this.phase !== 'starting' && plain(this.model?.progress?.text)) || T.starting;
  }
  /** The answer, part by part (the panel's testReadout): its root carries the engine's verdict and where the test was
   *  not measured (data-caveats: their codes), each part shows when the verdict has it, with its hooks (data-part;
   *  data-word yes | no | fewer | unclear; the fewer part's data-why and, for the one sentence, data-shrink; the cost
   *  rows' data-row and the cost part's data-shrink when it says the cells shrank; each went bar's parts by data-went, as
   *  wide as their whole percentages); after Run again, the earlier answer's word and why a repeat is no stronger
   *  answer. Where it was not measured is said in tooltips only: the fewer part's ⓘ, else the back part's own, or the
   *  too-few line's. */
  renderAnswer(r) {
    setData(this.answerBox, 'verdict', typeof r?.verdict === 'string' && r.verdict ? r.verdict : null);
    setData(this.answerBox, 'caveats', r ? codesOf(r.caveats) : null);
    const repeat = r?.repeat && typeof r.repeat === 'object' ? r.repeat : null;
    setText(this.repeatText, plain(repeat?.text)); setText(this.repeatNote, plain(repeat?.note)); setHidden(this.repeatNode, !plain(repeat?.text));
    const parts = this.answerParts, part = key => r?.[key] && typeof r[key] === 'object' ? r[key] : null;
    const back = part('back'), fewer = part('fewer'), cost = part('cost'), went = part('went');
    setHidden(parts.back.root, !back);
    if (back) {
      setText(parts.back.title, plain(back.title)); setText(parts.back.word, plain(back.answer)); setData(parts.back.root, 'word', plain(back.word) || null);
      setText(parts.back.text, plain(back.text)); setTitle(parts.back.text, plain(back.tip));
      const info = plain(back.info);
      setTitle(parts.back.info, info); setHidden(parts.back.info, !info);
    }
    setHidden(parts.fewer.root, !fewer);
    if (fewer) {
      setText(parts.fewer.title, plain(fewer.title)); setText(parts.fewer.verdict, plain(fewer.head));
      setText(parts.fewer.text, plain(fewer.text)); setText(parts.fewer.why, plain(fewer.whyText));
      setData(parts.fewer.root, 'word', plain(fewer.word) || null); setData(parts.fewer.root, 'why', plain(fewer.why) || null);
      setData(parts.fewer.root, 'shrink', fewer.shrink === true ? '' : null);
      // Its ⓘ's label names what its tooltip holds: how often the test is wrong, where it was not measured, or both.
      const tip = plain(fewer.tip), caveats = textsOf(r.caveats).length > 0;
      const label = !caveats ? T.aboutLabel : plain(r.calibration) ? T.aboutCaveatsLabel : T.caveatsLabel;
      setTitle(parts.fewer.info, tip); setHidden(parts.fewer.info, !tip);
      if (parts.fewer.info.getAttribute('aria-label') !== label) parts.fewer.info.setAttribute('aria-label', label);
    }
    setHidden(parts.cost.root, !cost);
    if (cost) {
      setText(parts.cost.title, plain(cost.title));
      const rows = new Map((Array.isArray(cost.rows) ? cost.rows : []).map(x => [x?.key, x]));
      for (const key of COST_ROWS) {
        const row = parts.cost.rows[key], x = rows.get(key);
        setHidden(row.root, !x);
        if (x) { setText(row.label, plain(x.label)); setText(row.value, plain(x.value)); setText(row.change, plain(x.change)); }
      }
      const said = cost.shrink === true && !!plain(cost.text);
      setText(parts.cost.shrink, plain(cost.text)); setHidden(parts.cost.shrink, !said); setData(parts.cost.root, 'shrink', said ? '' : null);
    }
    setHidden(parts.went.root, !went);
    if (went) {
      setText(parts.went.title, plain(went.title));
      const sides = (Array.isArray(went.sides) ? went.sides : []).filter(s => s && ['A', 'B'].includes(s.side));
      for (const side of ['A', 'B']) {
        const bar = parts.went.bars[side], s = sides.find(x => x.side === side), percents = new Map((s?.parts ?? []).map(p => [p?.key, p?.percent]));
        setHidden(bar.root, !s);
        setText(bar.label, side === 'A' ? T.asNow : this.testName());
        setTitle(bar.bar, plain(s?.text)); if (bar.bar.getAttribute('aria-label') !== plain(s?.text)) bar.bar.setAttribute('aria-label', plain(s?.text));
        for (const key of WENT) {
          const width = `${finite(percents.get(key)) ? clamp(percents.get(key), 0, 100) : 0}%`;
          if (bar.segs[key].style.width !== width) bar.segs[key].style.width = width;
        }
      }
      // The parts' key, once per words: each part's colour and word.
      const first = sides[0]?.parts ?? [], key = first.map(p => `${p?.key}\u0000${plain(p?.label)}`).join('\u0001');
      if (key !== this.wentKey) {
        this.wentKey = key;
        parts.went.key.replaceChildren(...first.filter(p => WENT.includes(p?.key)).map(p => {
          const item = this.el('span', 'qc-tc-went-item'); item.dataset.went = p.key; item.append(this.el('i'), this.el('span', '', plain(p.label)));
          return item;
        }));
      }
    }
    for (const key of ['alone', 'toofew']) {
      const text = plain(part(key)?.text);
      setText(parts[key].text, text); setHidden(parts[key].root, !text);
    }
    setData(parts.toofew.root, 'why', plain(part('toofew')?.why) || null); setTitle(parts.toofew.text, plain(part('toofew')?.tip));
  }

  // ---- placing --------------------------------------------------------------------------------------------------------
  /** The answer beside the views on a stage wide enough for it (WIDE_STAGE), else under them; the views' size for the
   *  room the stage gives (two views side by side, and the answer's column when it is beside them; while watching as tall
   *  as everything around them allows, CHROME_PX, with the muted lines that show; at the answer VIEW_PX, the card's body
   *  scrolling where the stage is shorter), and the corner kept inside the stage. Call it when the stage changes size. */
  layout() {
    if (!this.isOpen) return;
    const { width: W, height: H } = this.stageSize();
    const done = this.phase === 'done', beside = done && !this.minimised && W >= WIDE_STAGE;
    this.setBeside(beside);
    const room = W - 2 * MARGIN - FRAME_PX - VIEW_GAP - (beside ? ANSWER_GAP + ANSWER_PX : 0);
    const chrome = CHROME_PX.watch + this.lineCount() * CHROME_PX.line;
    const wide = W > 0 ? Math.floor(room / 2) : VIEW_PX;
    // Watching, a short stage gets smaller views; at the answer they keep VIEW_PX and the card's body scrolls.
    const tall = done || !(H > 0) ? VIEW_PX : Math.floor(H - 2 * MARGIN - chrome);
    const px = clamp(Math.min(wide, tall), MIN_VIEW_PX, VIEW_PX), dpr = clamp(globalThis.devicePixelRatio || 1, 1, 3);
    if (px !== this.viewPx || dpr !== this.dpr) { this.viewPx = px; this.dpr = dpr; this.invalidate(); }
    const device = Math.max(1, Math.round(px * this.dpr));
    for (const role of ROLES) {
      const p = this.parts[role];
      if (p.px === device) continue;
      p.px = device; p.canvas.width = device; p.canvas.height = device;
      p.canvas.style.width = p.canvas.style.height = `${px}px`;   // its frame wraps it (with a 1 px border)
      p.fig.style.width = `${px + 2}px`;                            // the caption and numbers under it as wide as the view
      this.dirty[role] = true;
    }
    // Beside the answer, the views' column is as wide as the views: its lines wrap there, never pushing the answer.
    const column = beside ? `${2 * (px + 2) + VIEW_GAP}px` : '';
    if (this.watchBox.style.width !== column) this.watchBox.style.width = column;
    this.moveTo(this.left, this.bottom);
  }
  /** The answer beside the views (the card's `qc-tc-wide`: wider, shorter) or under them. */
  setBeside(on) {
    if (on === this.beside) return;
    this.beside = on; this.size = null;   // its usual size until the browser reports the new one
    this.root.classList?.toggle('qc-tc-wide', on);
  }
  /** How many muted lines show under the views (the suspects that couldn't be told apart, the views that may be older,
   *  the runs started again): each takes CHROME_PX.line. */
  lineCount() {
    const l = this.shownLines;
    return (l.views ? 1 : 0) + (l.stale ? 1 : 0) + (l.retry ? 1 : 0);
  }
  /** A muted line came or went: the views make room for it. */
  setLine(which, on) {
    if (this.shownLines[which] === on) return;
    this.shownLines[which] = on;
    this.layout();
  }
  /** The corner's size: as the browser reports it, else the phase's usual one with the muted lines that show, never
   *  taller than the stage leaves it (the card's body scrolls then). */
  extent() {
    if (this.size) return this.size;
    const phase = this.phase, kind = this.minimised ? 'small' : phase === 'form' ? 'form' : phase === 'waiting' ? 'waiting'
      : ENDED.has(phase) ? 'ended' : phase === 'done' ? (this.beside ? 'wide' : 'done') : 'watch';
    const [w, h] = SIZES[kind], { height: H } = this.stageSize();
    const tall = h + (kind === 'small' || !VIEWING.has(phase) ? 0 : this.lineCount() * CHROME_PX.line);
    return { w, h: H > 0 ? Math.min(tall, Math.max(0, H - 2 * MARGIN)) : tall };
  }
  /** To `left` and `bottom` (CSS px from the stage's bottom left), kept inside the stage (its header always). */
  moveTo(left, bottom) {
    const { width: W, height: H } = this.stageSize(), { w, h } = this.extent();
    this.left = Math.round(W > 0 ? clamp(left, MARGIN, Math.max(MARGIN, W - w - MARGIN)) : Math.max(MARGIN, left));
    this.bottom = Math.round(H > 0 ? clamp(bottom, MARGIN, Math.max(MARGIN, H - h - MARGIN)) : Math.max(MARGIN, bottom));
    const s = this.root.style;
    if (s.left !== `${this.left}px`) s.left = `${this.left}px`;
    if (s.bottom !== `${this.bottom}px`) s.bottom = `${this.bottom}px`;
    this.report();
  }
  /** Where the corner is ({left, top, bottom, w, h}, CSS px in the stage), or null while closed. */
  box() {
    if (!this.isOpen) return null;
    const { height: H } = this.stageSize(), { w, h } = this.extent();
    return { left: this.left, top: H > 0 ? Math.max(0, Math.round(H - this.bottom - h)) : null, bottom: this.bottom, w: Math.round(w), h: Math.round(h) };
  }
  report() {
    const b = this.box();
    if (!b) return;
    const key = `${b.left},${b.bottom},${b.w},${b.h},${b.top}`;
    if (key === this.boxKey) return;
    this.boxKey = key; this.call('onBox', b);
  }

  // ---- painting -------------------------------------------------------------------------------------------------------
  painting() { return this.isOpen && !this.minimised && !this.hidden && VIEWING.has(this.phase); }
  glowsIn(role) { const fade = this.fades[role]; return (this.shown[role]?.lit ?? 0) > 0 || (fade?.from?.lit ?? 0) > 0; }
  glowing() { return !this.still && ROLES.some(role => this.glowsIn(role)); }
  /** One paint is due: at once for a change, else at the glow's next tick; never sooner than TICK_MS after the last. */
  wake() {
    if (!this.painting() || this.paintFrame != null || this.timers.has('paint')) return;
    if (!this.dirty.control && !this.dirty.test && !this.fades.control && !this.fades.test && !this.glowing()) return;
    this.later('paint', this.paintedAt + TICK_MS - this.now(), () => {
      this.paintFrame = this.clock.requestAnimationFrame(() => { this.paintFrame = null; this.paint(); this.wake(); });
    });
  }
  /** Each view that changed is drawn whole (its frame, fading in over the one before, and the halos); a view that only
   *  breathes repaints just its glowing patch. */
  paint() {
    if (!this.painting()) return;
    const now = this.now(), still = this.still, level = still ? 1 : breath(now);
    let glowed = false;
    for (const role of ROLES) {
      const fade = this.fades[role], t = fade && !still ? fadeProgress(now - fade.start) : 1;
      const whole = this.dirty[role] || !!fade;
      if (!whole && (still || !this.glowsIn(role))) continue;
      this.dirty[role] = false;
      const lit = this.paintView(role, whole, fade && t < 1 ? fade.from : null, t, level);
      if (fade && t >= 1) this.fades[role] = null;
      this.draws++; this.hook?.count('qcTestCornerDraws');
      if (lit > 0 && !still) glowed = true;
    }
    this.paintedAt = now;
    if (glowed) { this.glowTimes.push(now); while (this.glowTimes[0] <= now - 1000) this.glowTimes.shift(); }
  }
  /** One view: its frame (over the one before while it fades in) and the frames' halo layers at the breath's level, or
   *  (`whole` false) only the patch the halos cover. Returns how many suspects glow in it. */
  paintView(role, whole, from, t, level) {
    const canvas = this.parts[role].canvas, ctx = canvas.getContext?.('2d');
    if (!ctx) return 0;
    const g = this.geometry(), to = this.shown[role];
    const toComp = to && g ? this.composite(to, g) : null, fromComp = from && g ? this.composite(from, g) : null;
    const toGlow = toComp ? this.glowLayer(to, g) : null, fromGlow = fromComp ? this.glowLayer(from, g) : null;
    if (!whole) {
      const b = toGlow ? to.glow.box : null;
      if (!b) return 0;
      ctx.save(); ctx.beginPath(); ctx.rect(b[0], b[1], b[2], b[3]); ctx.clip();
      ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(toComp, b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]);
      ctx.globalCompositeOperation = 'screen'; ctx.globalAlpha = level;
      ctx.drawImage(toGlow, b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]);
      ctx.restore();
      return to.lit;
    }
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    if (!toComp) { ctx.fillStyle = COLORS.background; ctx.fillRect(0, 0, canvas.width, canvas.height); return 0; }
    if (fromComp) { ctx.drawImage(fromComp, 0, 0); ctx.globalAlpha = t; }
    ctx.drawImage(toComp, 0, 0);
    if (fromGlow || toGlow) {
      ctx.globalCompositeOperation = 'screen';
      if (fromGlow) { ctx.globalAlpha = (1 - t) * level; ctx.drawImage(fromGlow, 0, 0); }
      if (toGlow) { ctx.globalAlpha = (fromComp ? t : 1) * level; ctx.drawImage(toGlow, 0, 0); }
      ctx.globalCompositeOperation = 'source-over';
    }
    ctx.globalAlpha = 1;
    return (toGlow ? to.lit : 0) + (fromGlow ? from.lit : 0);
  }
  geometry() {
    if (!this.crop) return null;
    const size = Math.max(1, Math.round(this.viewPx * this.dpr)), key = `${size}|${this.dpr}|${this.cropKey}`;
    if (this.geom?.key !== key) this.geom = { ...geometry(this.crop, size, this.dpr), key };
    return this.geom;
  }
  /** A frame drawn once (drawFrame, with its suspects' cores) on its own canvas and kept for repaints. */
  composite(frame, g) {
    if (frame.comp?.key === g.key) return frame.comp.canvas;
    const canvas = frame.comp?.canvas ?? this.doc.createElement('canvas');
    canvas.width = g.size; canvas.height = g.size;
    const ctx = canvas.getContext?.('2d');
    if (!ctx) return null;
    drawFrame(ctx, frame, g, this.suspects, this.colors(frame));
    drawCores(ctx, this.suspects, frame.cell != null ? frame.weights : null, g, this.sprite('core'));
    frame.comp = { key: g.key, canvas };
    this.remember(frame);
    return canvas;
  }
  /** Each cell's colour in a frame, by slot: the page's colour of the cell of that name in your run (colorOf, when the
   *  view names it), else the stage's identity colour of that cell, else (no name) one of its own. */
  colors(frame) {
    const parents = frame.parents, colorOf = typeof this.o.colorOf === 'function';
    return Array.from(frame.shape?.ids ?? [], label => {
      const parent = parents?.get(label);
      if (parent == null) return identityRGB(label);
      return (colorOf ? parseColor(this.call('colorOf', parent)) : null) ?? identityRGB(parent);
    });
  }
  /** A frame's halos at full strength (drawGlow on a clear canvas the view's size), kept like its drawing: a breath then
   *  draws them at once at the breath's level (two image draws a view, however many suspects). Null with none inside. */
  glowLayer(frame, g) {
    if (!(frame.lit > 0)) return null;
    if (frame.glow?.key === g.key) return frame.glow.canvas;
    const canvas = frame.glow?.canvas ?? this.doc.createElement('canvas');
    canvas.width = g.size; canvas.height = g.size;
    const ctx = canvas.getContext?.('2d');
    if (!ctx) return null;
    ctx.clearRect(0, 0, g.size, g.size);
    drawGlow(ctx, this.suspects, frame.weights, g, 1, this.sprite('halo'));
    frame.glow = { key: g.key, canvas, box: glowBox(this.suspects, frame.weights, g) };
    this.remember(frame);
    return canvas;
  }
  /** The frames with drawings kept, the oldest not on screen let go first (MAX_COMPOSITES). */
  remember(frame) {
    this.composites = this.composites.filter(f => f !== frame); this.composites.push(frame);
    const busy = new Set([this.shown.control, this.shown.test, this.fades.control?.from, this.fades.test?.from]);
    for (let i = 0; this.composites.length > MAX_COMPOSITES && i < this.composites.length;) {
      if (busy.has(this.composites[i])) { i++; continue; }
      const old = this.composites.splice(i, 1)[0]; old.comp = null; old.glow = null;
    }
  }
  /** The halo and the core, each drawn once: a soft radial falloff (COLORS.halo, COLORS.core). */
  sprite(kind) {
    if (kind in this.sprites) return this.sprites[kind];
    const c = this.doc.createElement?.('canvas'), ctx = c?.getContext?.('2d');
    if (!ctx || typeof ctx.createRadialGradient !== 'function') return (this.sprites[kind] = null);
    c.width = c.height = 64;
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    if (!grad) return (this.sprites[kind] = null);
    for (const [at, colour] of COLORS[kind]) grad.addColorStop(at, colour);
    ctx.fillStyle = grad; ctx.fillRect(0, 0, 64, 64);
    return (this.sprites[kind] = c);
  }

  // ---- the harness ------------------------------------------------------------------------------------------------------
  /** For the harness: what the corner shows now. `frames`: the frames kept (both runs); `shown`: per run
   *  the frame on screen (`kept` and `of`: "Suspects kept: kept of of"; `held_by` how its shares were placed; `unsure`:
   *  how many of its suspects it couldn't tell apart, left out of "of"); `glow.fps`: glow paints in the last second (0
   *  while still or off); `colors` how the other cells are coloured ('stage', 'own', null before a frame); `stale` the
   *  views at the answer could not be had (FINAL_TRIES asks): they may show an earlier iteration; `beside` the answer
   *  reads beside the views (a wide stage). */
  snapshot() {
    const shown = role => {
      const f = this.isOpen ? this.shown[role] : null;
      return f ? { seq: f.seq, iter: f.iter, voxel: f.voxel, kept: f.cell == null ? null : f.kept, of: f.cell == null ? null : f.of,
        held_by: f.heldBy, cell_total: f.cell == null ? null : f.cellTotal, found: f.cell != null, unsure: f.unsureCount } : null;
    };
    const now = this.now(), on = this.painting() && ROLES.some(role => (this.shown[role]?.lit ?? 0) > 0);
    return {
      open: this.isOpen, phase: this.isOpen ? this.phase : 'closed', minimised: this.minimised,
      frames: this.frames.control.length + this.frames.test.length, kept: { control: this.frames.control.length, test: this.frames.test.length },
      steps: this.steps.length, scrub: this.scrub, suspects: this.suspects.length, shown: { control: shown('control'), test: shown('test') },
      glow: { on, fps: on && !this.still ? this.glowTimes.filter(t => t > now - 1000).length : 0, still: this.still },
      draws: this.draws, fetches: this.fetches, place: this.box(), colors: this.isOpen ? this.colouring() : null,
      stale: this.isOpen && this.stale, beside: this.isOpen && this.beside,
    };
  }
  /** Every word the corner shows now (text, tooltips and labels), for the harness's banned-word check. */
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
  /** Every string the corner can show (its deck's; tests). */
  static allTexts() { return allTexts(); }
  dispose() {
    this.close();
    this.demo?.dispose();
    for (const key of [...this.timers.keys()]) this.cancel(key);
    for (const [node, type, fn, opts] of this.listeners) node?.removeEventListener?.(type, fn, opts);
    this.listeners = []; this.observer?.disconnect?.(); this.root.remove?.();
  }
}

/** A string from the panel's model, trimmed ('' for anything else). */
function plain(v) { return typeof v === 'string' ? v.trim() : ''; }
/** The settings the panel offers in the choice ({flag, name, now, nowText, nowTip, tried, note, tip}, those with a flag). */
const listOf = settings => (Array.isArray(settings) ? settings : []).filter(s => s && typeof s.flag === 'string');
/** The panel's reasons ([{code, text}]) as their words, and as their codes for a hook (a list of codes too:
 *  "not_at_end check_off"; '' for none). */
const textsOf = list => (Array.isArray(list) ? list : []).map(r => plain(r?.text)).filter(Boolean);
const codesOf = list => (Array.isArray(list) ? list : []).map(r => typeof r === 'string' ? r : r?.code)
  .filter(code => typeof code === 'string' && code).join(' ');
function setText(node, text) { text = text == null ? '' : String(text); if (node && node.textContent !== text) node.textContent = text; }
function setHidden(node, hidden) { if (node && node.hidden !== hidden) node.hidden = hidden; }
function setTitle(node, title) { title = title ?? ''; if (node && node.title !== title) node.title = title; }
/** A data attribute set to `value`, or taken away for null (a hook that is there only while it holds). */
function setData(node, key, value) {
  if (!node) return;
  if (value == null) { if (key in node.dataset) delete node.dataset[key]; } else if (node.dataset[key] !== value) node.dataset[key] = value;
}
