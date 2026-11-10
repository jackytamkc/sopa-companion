// Go deeper: one cell's RNA, explained. A movable window inside the
// stage (never a <dialog>: an open dialog disables the app's keys) that shows, for the selected cell, how the mixing
// check's model splits its RNA into its own, from neighbours and loose RNA in the tissue, and which genes it puts where.
//
// Which result: always the newest one at or before the frame on screen, never a later one (the check's rule); the header
// names its iteration. A frame without proseg's count of its own (between counted frames, or its file gone) is answered
// as the counted frame the engine snaps it to, and the header says so. While a run is live the engine follows the
// window's cell (POST /runs/{id}/qc/deeper follow: true) and its results ride the analysed passes ({ev: "deeper"}
// pushes); the window applies at most one a second, in the paint batch of the next snapshot. Paused, stepped or scrubbed
// frames are asked for (150 ms debounce); a cached frame is answered from memory. Failures are asked again after 2, 5,
// 10 and 30 s.
//
// Motion is change-driven, the check's rules: only for a change at least MOTION_GAP_MS after the previous one, at most 4
// paints per update on the stage's 150 / 300 / 450 ms steps (the headline counts, the bar glides, a new gene chip slides
// in with one pulse, rows slide when their rank changes, the small line's newest segment draws on); nothing moves while
// nothing changes or while paused (apart from the one transition a scrub or step causes), and reduced motion or the
// check's calm motion lands every change in one paint. Under Compare's full-screen view (setCovered) nothing moves or
// changes; the window catches up once it closes. No CSS animation or transition, no spinner: "Working out…" shows only
// after WORKING_MS.
//
// The window is built once and updated in place; a section holding a pressed control or keyboard focus is never rebuilt
// (the change follows once it is let go). Every word comes from qc-deeper-copy.mjs (or the check's own deck): on screen
// a number, a few-word verdict and labels, the explanations in tooltips.
//
// "Show on the image", under the source rows: the transcripts proseg counts for the cell that likely came from
// neighbours glow on the stage (suspectScores: P = w x pi, w proseg's share of the transcript for the cell from the where
// reply, exact from proseg's own pools or approximate from the frame's layers, pi the model's per-gene split placed by
// position, a rough guide; Stage.setGlow through onGlow, in 2-D and 3-D, there at each transcript's depth), some just
// outside the drawn outline where proseg moved them in; the chips' genes as dots at their transcripts' places (GET
// /runs/{id}/qc/deeper/where for the result's own frame; 2-D), every other cell faded, the camera moved once to the cell
// and its sources (in 3-D keeping the user's angle); "Where they sit" says, per source, where each gene's transcripts
// lie, all on the count the reply read (averaged, or the count proseg saves at a pause, the hold, a finish or the end:
// said so). Following a cell as the run plays, an approximate reply right after an exact one of the counted frame
// before it, for the same rows and genes, keeps the exact one on screen, said so; once the run stops playing the frame
// is asked for itself, and again when it rests there (proseg's exact shares of a paused frame). A row's Show glows only
// that neighbour's suspects and fits the camera to the cell and that neighbour. When none of the glowing transcripts
// lies near the neighbour the glow would name (sourceSupport), that neighbour is not shown as their source: the line
// says they look like a cell with no outline here, the glow has no side to ripple in from and the camera fits the cell
// and its glow. Off by default; closing the window, another cell or another run turns it off. While on it follows the
// result shown: asked again at most once a second live, once per frame paused.
//
// While the window shows a cell (2-D and 3-D), only the cells involved stay lit (its focus: the cell and its shown rows,
// lit, dim 1; the stage draws the cell clearest, its rows second); the basis beside the small line is proseg's count of
// the cell, named by which count the answer reads, with what proseg reads as background in its tooltip (context
// cellStats). With the view on, the cell's starting outline is drawn dashed on the stage in 2-D (onStartOutline; the
// glow's tooltip names it only there), and the start line's own reply tells where each glowing transcript sat at the
// start (startPlaces: inside its own starting outline, where no cell started, inside its neighbour's or another
// cell's), said in one line, with the start's hold of those inside its own outline in the line's tooltip (a detail
// run's start layers: context startHold); the camera's move keeps clear of the window and of the test corner. Beside Show on the image, "Test a
// setting here…" (while the page can test this cell: context testable) opens the mixing check's test of a setting on
// this cell (onTest); with the check off it sits beside the check's switch instead (a press of its Run switches the
// check on first: the test starts from its suspects).
import {
  DEEP_TEXT as T, deep, headlinePercent, stepKey, startChange, barLabels, chipText, rowTooltip, sharePct, foldText,
  sameKindCountText, partlyText, genesList, howItems, frameTag, cantRunText, detailFailedText, count, allTexts,
  lookAlikeLine, partPercent, startLine, startTip, glowTip, noOutlineLine, noOutlineHead, START_BAND, underFloor, LIKE_NOW,
} from './qc-deeper-copy.mjs';
import { QC_TEXT, copy as qcCopy, number, duration, reasonText, startName, cellShare, unavailableTooltip } from './qc-copy.mjs';
import { clusterColorCSS } from './clusters.js';

export const RESULT_VERSION = 1;
export const APPLY_MS = 1000;          // live: at most one update a second
export const WITH_SNAPSHOT_MS = 1000;  // a pushed result waits at most this long for a snapshot's paint batch
export const MOTION_GAP_MS = 900;      // motion only for a change at least this long after the previous one
export const STEP_MS = 150;            // the stage's motion steps (150 / 300 / 450 ms)
export const NUMBER_MS = 450;          // the headline counts to its new value
export const GLIDE_MS = 300;           // bars glide, chips slide, rows move
export const MAX_PAINTS = 4;           // per update, the first one with the snapshot
export const MOVE_WINDOW_MS = 2000;    // a scrub or step: its answer within this long is still its transition
export const WORKING_MS = 250;         // "Working out…" only after this long
export const ASK_MS = 150;             // paused, stepped or scrubbed frames are asked for after this debounce
export const RETRY_MS = Object.freeze([2000, 5000, 10000, 30000]);
export const STALE_MS = 15000;         // an answer promised by the engine ("working") is asked again after this long
export const FOLLOW_RENEW_MS = 30000;  // while open on a live run the engine's follow is asked again this often (a lease:
                                       // engine/analysis.py DEEPER_FOLLOW_TTL_S ends one nobody renews)
export const GROUPS_LAG = 50;          // cell groups this many iterations from the result's (or newer) are named on screen
export const HISTORY_KEPT_POINTS = 2000;   // a cell's small line keeps at most this many points
export const RANK_RATIO = 1.25;        // rows and chips change rank only when a value passes 1.25 x the one above
// Fewer words (about 95 for a typical ready cell): at most 4 chips (then "+k more"), a row's second line
// at most 2 genes; its own top genes ride in its own kind's tooltip (the table holds every gene).
export const CHIPS = 4, ROWS = 5, ROW_MIN = 0.01, SOURCES_MIN = 5, OWN_GENES = 3, KIND_GENES = 4, TOO_FEW_GENES = 8;
export const SOURCE_GENES = 2;         // a row's second line: the genes it seems to give, at most this many
export const FEW_TX = 200, FLOOR_CODE = 4, SAME_KIND = 0.9;
export const SPLIT_MIN = 100;           // the worker splits a cell from this many transcripts on ("at least 100")
export const ROW_PITCH = 64;           // a source row's height + gap (CSS px: the slide needs no layout read)
export const LIT_MAX = 6;              // the focus's lit cells: the cell and its rows (the stage's SPILL_LIT)
export const TABLE_ROW_PX = 30, TABLE_ROWS = 12, TABLE_PAD = 6;
export const KEEP_RESULTS = 64, KEEP_CELLS = 3, HISTORY_POINTS = 200;
export const SNAPPED_KEPT = 512;       // frames whose answer the engine gave as another counted frame's, remembered
export const WIDTH = 360, RIGHT = 12, TOP = 52, MARGIN = 8, SHORT_STAGE = 480;
export const STORAGE_KEY = 'sopa-companion:qc-deeper:place';
export const COLORS = Object.freeze({ own: '#3987e5', nb: '#d55181', loose: '#b8c0cc', tag: '#f08cbc', same: '#9aa4b2' });
export const COUNTERS = Object.freeze(['qcDeeperOpened', 'qcDeeperApplied', 'qcDeeperAnimFrames', 'qcDeeperRetargets',
  'qcDeeperWhereOn', 'qcDeeperWhereAsked', 'qcDeeperWhereShown', 'qcDeeperWhereStartAsked', 'qcDeeperWhereSnapped']);
// "Show where on the image": the chips' genes (at most WHERE_GENES) and this cell's own genes found inside its sources (at
// most WHERE_OWN). Colours in a fixed order, a gene keeping its colour while the toggle is on (assignSlots): Okabe-Ito and
// Tol hues stepped for the dark stage, never magenta (from neighbours), the own blue or white (the cell). The first four
// stay apart for protan and deutan readers too (OKLab ΔE x 100 under Machado 2009: >= 11.5 between them, >= 8.4 from those
// three), the fifth and sixth for full colour vision only, so every gene also names itself (its chip wears its colour, its
// line in "Where they sit" too). Own genes in the own blue family.
export const WHERE_GENES = 6, WHERE_OWN = 3;
export const WHERE_REFRESH_MS = 1000;            // live: asked again at most once a second
export const WHERE_WAIT_MS = 1000, WHERE_WAITS = 5;   // a frame not saved yet (409): asked again this often, this many times
export const WHERE_COLORS = Object.freeze({ from: Object.freeze(['#f0e442', '#44bb99', '#e8622c', '#ff9f1c', '#ff9ecb', '#7bd148']),
  own: Object.freeze(['#3987e5', '#8ec5ff', '#2350c8']) });
// A dot's opacity: a chips' gene in the cell or a source, or this cell's own gene inside a source (lit); a chips' gene in
// the tissue around (around); this cell's own gene in the cell or around, where it is expected (expected).
export const WHERE_ALPHA = Object.freeze({ lit: 1, around: 0.5, expected: 0.35 });
// The engine's word per gene and its words on screen (its thresholds are a rough guide).
export const WHERE_WORDS = Object.freeze({ edge: ['deepWhereEdge', 'deepWhereEdgeTooltip'], spread: ['deepWhereSpread', 'deepWhereSpreadTooltip'],
  layer: ['deepWhereLayer', 'deepWhereLayerTooltip'], few: ['deepWhereFew', 'deepWhereFewTooltip'] });
// "few" under this many in proseg's count of the gene, or placed of it where the shares are approximate (the engine's
// FEW_IN_CELL): a gene proseg counts this many of or more is "few" for want of placed ones, and says so.
export const WHERE_FEW = 4;
export const WHERE_MARGIN = 12;                  // CSS px kept clear around the camera's box and beside the window
export const WHERE_V = Object.freeze([3]);       // the where replies taken (v 3: points.w, proseg's share, and held)
export const HELD = Object.freeze(['exact', 'approx', 'none']);   // where the reply's shares came from
// The suspects' glow: a transcript glows at P = w x pi >= GLOW_MIN, brighter the likelier, whoever's outline it lies in;
// "likely" at pi >= LIKELY_MIN where proseg counts at least LIKELY_W of it for the cell; a test of a setting starts with
// the top SUSPECTS_SENT by P (the engine takes at most 400).
export const GLOW_MIN = 0.2, LIKELY_MIN = 0.5, SUSPECTS_SENT = 400;
export const LIKELY_W = 0.5;
export const PAIR_MARGIN_UM = 6;                 // a row's Show: the camera fits the cell and that neighbour, this much around
export const MATCH_UM = 0.01;                    // the start line's reply: a transcript's gene and x, y within this
// Where a glowing transcript sat at the start (startPlaces' codes: PLACES[code - 1]; 0 for a point proseg does not
// count for the cell): inside its own starting outline, where no cell started, inside the starting outline of the neighbour it
// seems to come from (its giver), inside another cell's, or not in the start line's reply. The engine's own classes for
// the test's pool (engine/qc_pool.py Item.start_class: OWN, NONE, GIVER_*, OTHER) by the same words.
export const PLACES = Object.freeze(['own', 'none', 'giver', 'other', 'unknown']);
// The start's hold of a spot (context startHold: PriorLayers.startHold) as the line's tooltip splits it.
export const HOLD = Object.freeze({ unknown: 0, nucleus: 1, cell: 2 });
const HOLD_NAMES = Object.freeze(['unknown', 'nucleus', 'cell']);
export const SIDE_MIN = 0.45;                    // the camera's move keeps clear of the window on a side at least this share
                                                 // of the stage wide, of the test corner where the box still shows at least
                                                 // this share as large (else either may cover a little of the box)
// Inline icons: one stroke weight, drawn in the text's colour; static markup only, never data.
export const ICONS = Object.freeze({
  info: '<circle cx="8" cy="8" r="6.25"/><path d="M8 7.25v3.75M8 5h.01"/>',
  show: '<path d="M1.75 8S4 3.75 8 3.75 14.25 8 14.25 8 12 12.25 8 12.25 1.75 8 1.75 8z"/><circle cx="8" cy="8" r="1.9"/>',
  arrow: '<path d="M6 3.5 10.5 8 6 12.5"/>',
  close: '<path d="M4 4l8 8M12 4l-8 8"/>',
  minimise: '<path d="M3.5 8h9"/>',
  expand: '<path d="M3.5 8h9M8 3.5v9"/>',
  find: '<circle cx="8" cy="8" r="4.25"/><path d="M8 1.5V4M8 12v2.5M1.5 8H4M12 8h2.5"/>',
});
const svg = name => `<svg viewBox="0 0 16 16" width="14" height="14" focusable="false">${ICONS[name]}</svg>`;
const WHERE_ROLES = new Set(['from', 'own']);
const OWNER_CELL = 1;                            // the reply's owner classes: 0 none, 1 the cell, 2 + k source k, 255 another cell
const CHECK_STATES = new Set(['off', 'waiting', 'learning', 'failed', 'unavailable']);
const RESULT_STATES = new Set(['ready', 'too_few', 'empty', 'not_checked']);
const READY_VIEWS = new Set(['waiting-start', 'live', 'paused', 'scrubbing', 'finished']);
// Proseg gone, or writing its results (finishing: the engine refuses a pause and a where export then): the run can be
// neither paused nor asked for a share.
const RUN_ENDED = new Set(['done', 'stopped', 'failed', 'finishing']);
const finite = Number.isFinite;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ---- pure helpers (node tests) ------------------------------------------------------------------------------------
/** The key of the newest result at or before frame `seq` (0: the start line, 'start'), never a later one; null when
 *  there is none. `keys`: the results held ('start' and frame seqs). */
export function newestAtOrBefore(keys, seq) {
  if (seq === 0 || seq === 'start') { for (const k of keys) if (k === 'start') return 'start'; return null; }
  if (!Number.isInteger(seq) || seq < 0) return null;
  let best = null;
  for (const k of keys) if (Number.isInteger(k) && k <= seq && (best == null || k > best)) best = k;
  return best;
}
/** Rank `items` (keys) by `value(key)`, starting from the order shown before (`previous`): an item moves above the one
 *  over it only when its value passes `ratio` x that one's (so ranks do not flicker between near-equal values); new
 *  items enter where their value puts them. */
export function rankOrder(previous, items, value, ratio = RANK_RATIO) {
  const wanted = new Set(items), order = (previous || []).filter(k => wanted.has(k));
  const known = new Set(order);
  for (const k of [...items].filter(k => !known.has(k)).sort((a, b) => value(b) - value(a))) {
    let i = order.length; while (i > 0 && value(order[i - 1]) < value(k)) i--;
    order.splice(i, 0, k);
  }
  for (let pass = 0, moved = true; moved && pass < order.length; pass++) {
    moved = false;
    for (let i = 1; i < order.length; i++) if (value(order[i]) > ratio * value(order[i - 1])) { [order[i - 1], order[i]] = [order[i], order[i - 1]]; moved = true; }
  }
  return order;
}
/** Progress (0-1, eased) of a motion of `duration` ms at `elapsed` ms: the first paint (with the snapshot) already
 *  moves one step, so a 450 ms count shows 1/4, 1/2, 3/4 and 1 on its four paints. */
export function motionProgress(elapsed, duration) {
  const t = clamp((elapsed + STEP_MS) / (duration + STEP_MS), 0, 1);
  return 1 - (1 - t) ** 3;
}

const num = (v, d = 0) => (finite(v) && v >= 0 ? v : d);
/** A v1 result (6.5) checked and normalised; null when it is not one. Lenient about extra fields. */
export function normalise(raw) {
  if (!raw || typeof raw !== 'object' || !RESULT_STATES.has(raw.state)) return null;
  if (raw.v != null && raw.v !== RESULT_VERSION) return null;
  const seq = raw.seq === 'start' ? 'start' : Number.isInteger(raw.seq) && raw.seq >= 0 ? raw.seq : null;
  if (seq == null) return null;
  // Which of proseg's counts it reads (the engine's count_of the frame): "sampled", the count proseg saves at a pause,
  // the hold, a finish or the end (one random background draw); else (absent too) "averaged" over that draw.
  const r = { v: RESULT_VERSION, state: raw.state, seq, iter: finite(raw.iter) ? raw.iter : null, cell: raw.cell, model: raw.model ?? null,
    count: raw.count === 'sampled' ? 'sampled' : 'averaged',
    transcripts: num(raw.transcripts), outside: num(raw.outside), code: Number.isInteger(raw.code) ? raw.code : null, ms: finite(raw.ms) ? raw.ms : null };
  if (raw.state === 'too_few') {
    r.genes = (Array.isArray(raw.genes) ? raw.genes : []).filter(g => Array.isArray(g) && typeof g[0] === 'string').map(g => [g[0], num(g[1])]);
    return r;
  }
  if (raw.state !== 'ready') return r;
  if (!Number.isInteger(r.code) || r.code < 0 || r.code > 250) return null;
  const parts = Array.isArray(raw.parts) && raw.parts.length === 3 && raw.parts.every(finite) ? raw.parts.map(v => Math.max(0, v)) : null;
  r.parts = parts ?? [1 - cellShare(r.code), cellShare(r.code), 0];
  r.rough = !!raw.rough;
  r.genes = (Array.isArray(raw.genes) ? raw.genes : []).filter(g => Array.isArray(g) && typeof g[0] === 'string' && finite(g[1]))
    .map(g => ({ gene: g[0], tx: num(g[1]), own: num(g[2]), nb: num(g[3]), flag: g[4] === 1 || g[4] === true, giver: Number.isInteger(g[5]) ? g[5] : null }))
    .map(g => ({ ...g, loose: Math.max(0, g.tx - g.own - g.nb) }));
  r.kind = (Array.isArray(raw.kind) ? raw.kind : []).filter(g => typeof g === 'string');
  r.own_genes = (Array.isArray(raw.own_genes) ? raw.own_genes : []).filter(g => Array.isArray(g) && typeof g[0] === 'string').map(g => [g[0], num(g[1]), num(g[2])]);
  r.same_kind_share = finite(raw.same_kind_share) ? clamp(raw.same_kind_share, 0, 1) : 0;
  r.neighbours = (Array.isArray(raw.neighbours) ? raw.neighbours : []).filter(n => n && typeof n === 'object').map(n => ({
    cell: n.cell, um: finite(n.um) ? n.um : null, tx: num(n.tx),
    genes: (Array.isArray(n.genes) ? n.genes : []).filter(g => Array.isArray(g) && typeof g[0] === 'string').map(g => [g[0], num(g[1])]),
    kind: (Array.isArray(n.kind) ? n.kind : []).filter(g => typeof g === 'string'),
    // The worker's own class wins (it keeps a neighbour's class near the cut-off steady).
    same: typeof n.same === 'boolean' ? n.same : n.similar === true || (finite(n.similar) && n.similar >= SAME_KIND),
    similar: n.similar, known: n.known !== false,
  }));
  r.learned = ['yes', 'partly', 'no'].includes(raw.learned) ? raw.learned : 'yes';
  // The cell against itself at the start, like for like (the worker's start_change: {points, band, draws, now_tx,
  // start_tx, direction: its word, kept near the band as last shown, held, now and start: its share from neighbours now
  // and at the start in percent, both at the matched depth}), or null; a reply without a band reads the worker's
  // (START_BAND); one without now and start (an answer kept from before them) has none (null). Twin of
  // engine/qc_scoring.py normalise.
  const sc = raw.start_change;
  r.start_change = sc && typeof sc === 'object' && finite(sc.points) ? { points: sc.points, band: finite(sc.band) ? sc.band : START_BAND,
    draws: finite(sc.draws) ? sc.draws : null, now_tx: finite(sc.now_tx) ? sc.now_tx : null, start_tx: finite(sc.start_tx) ? sc.start_tx : null,
    direction: ['more', 'fewer', 'same'].includes(sc.direction) ? sc.direction : null, held: sc.held === true,
    now: finite(sc.now) ? sc.now : null, start: finite(sc.start) ? sc.start : null } : null;
  // A reply past the worker's 32 KB folds its 1-transcript genes into genes_rest [genes, transcripts, own, nb].
  const rest = Array.isArray(raw.genes_rest) && raw.genes_rest.length === 4 && raw.genes_rest.every(finite) ? raw.genes_rest : null;
  r.rest = rest && rest[0] > 0 ? { k: rest[0], tx: rest[1], own: rest[2], nb: rest[3] } : null;
  r.genes_total = Number.isInteger(raw.genes_total) && raw.genes_total >= r.genes.length ? raw.genes_total : r.genes.length + (r.rest?.k ?? 0);
  return r;
}

/** Whether the run has no start line, and why (only a start from cells alone has one: proseg's count at iteration 0):
 *  'nuclei' for a start from nuclei only (`startSource` "nuclei", or the engine's start kind "nuclei"), 'paired' for
 *  cells with their nuclei (start kind "paired": proseg's count at iteration 0 holds only the nuclei), '' for another
 *  kind but "cells" (the engine's refusal without one: 'none'); null while it has one, or nothing says otherwise
 *  (`startKind` null: not known yet). */
export function noStartLine(startSource = null, startKind = null) {
  if (startSource === 'nuclei' || startKind === 'nuclei') return 'nuclei';
  if (typeof startKind !== 'string' || startKind === 'cells') return null;
  return startKind === 'paired' ? 'paired' : '';
}
/** The window's words for frame 0 of a run without a start line (noStartLine's kind) while the page's Start and Step
 *  both work (context buttons, main.js runButtons: the run paused on its prior), which move it on; NO_START_ENDED_TEXT's,
 *  naming no button, everywhere else: a run that ended, is finishing or samples on by itself, one still pausing, and
 *  one paused later with its first frame scrubbed back to (its run button says Resume there, which leaves the view on
 *  that frame). */
export const NO_START_TEXT = Object.freeze({ nuclei: 'deepNucleiOnlyRun', paired: 'deepNucleiCellsRun', '': 'deepNoStartRun' });
export const NO_START_ENDED_TEXT = Object.freeze({ nuclei: 'deepNucleiOnlyRunEnded', paired: 'deepNucleiCellsRunEnded',
  '': 'deepNoStartRunEnded' });
/** Whether the run is paused on its prior as the page's run button tells it (main.js runButtons: Start, with Step
 *  beside it): proseg acknowledged the pause, and the run's first frame is its only one. The window reads the page's
 *  own buttons (context buttons); this is their rule for the prior, kept for the tests that hold the two together. */
export const onPrior = (run, frames) => run?.state === 'paused' && run.ack?.state === 'paused' && frames?.length === 1;

/** Everything the window shows for result `r` (normalised), from the deck: pure, so the rules can be tested. Short
 *  lines carry their explanation as a tooltip ({text, tip}). lookups: name(cellField) -> the cell's name,
 *  label(cellField) -> its label (1-based, 0 unknown), group(label) -> {label, color, iter} | null (the live cell
 *  groups), startCode (the cell's byte at the start: null unknown, 255 not judged), start (the result is the start
 *  line's), startSource and startKind (how the run's cells started: no change since the start without a start line,
 *  noStartLine), kept (the run's constants kept from other runs' measurement, Run.public constants.kept: "start_band"
 *  says so in the steps' tooltip). */
export function explain(r, { name = c => String(c), label = () => 0, group = () => null, startCode = null, start = false,
  startSource = 'xenium', startKind = null, kept = [] } = {}) {
  if (r.state === 'too_few') {
    const genes = [...r.genes].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, TOO_FEW_GENES);
    // At the start, the start line's own count, said as such; a count under the 100 never reads 100 (underFloor).
    const atStart = r.seq === 'start', n = underFloor(r.transcripts, SPLIT_MIN);
    return { state: 'too_few', lines: [atStart ? deep('deepTooFewStart', { n, start: startName(startSource) }) : deep('deepTooFew', { n }),
    ...(genes.length ? [deep('deepTooFewGenes', { genes: genes.map(([g, n]) => deep('deepOwnGene', { gene: g, n: count(n) })).join(' · ') })] : []),
    atStart ? T.deepTooFewLaterStart : T.deepTooFewLater] };
  }
  if (r.state === 'empty') return { state: 'empty', lines: [T.deepEmpty] };
  if (r.state === 'not_checked') return { state: 'not_checked', lines: [T.deepNotChecked] };
  const L = r.transcripts, code = r.code, p = headlinePercent(code), bar = barLabels(code, r.parts), step = stepKey(code);
  const nbInt = Math.floor(cellShare(code) * 100 + 1e-9);
  // The change since the start, like for like (the worker's start_change), and said so on the line; none for a run
  // without a start line (nuclei only, or cells with their nuclei: no starting number to compare with, as the panel says).
  // At a rest frame (an answer on the count proseg saves: a pause, the hold, a finish or the end) it is that iteration's
  // own word, with its own two numbers, as the panel shows a resting run's own change; never a word held from earlier
  // answers.
  const change = start || noStartLine(startSource, startKind) != null ? null
    : startChange(code, startCode, r.start_change ?? null, { rest: r.count === 'sampled' });
  // With the comparison's own two numbers (the answer's start_change now and start) the line says what it compares; a
  // word held from earlier answers, the same words without them.
  const stepLine = !change?.text ? T[step] : change.at ? deep('deepStepLikeAt', { step: T[step], change: change.text, ...change.at })
    : change.held ? deep('deepStepLikeHeld', { step: T[step], change: change.text })
      : deep(change.like ? 'deepStepLike' : 'deepStepChange', { step: T[step], change: change.text });
  // Its tooltip; with the two numbers on the line, what they are; a word held from earlier answers says so, with what
  // this answer reads on its own (startChange's `own`); with a like-for-like change on the line, that it reads the
  // averaged count on both sides when the answer reads the count as saved, and that its band was not measured again on
  // runs like this one (start_band kept).
  const stepTip = [T.deepStepsTooltip, change?.at ? T.deepLikeNumbersTooltip : '',
    LIKE_NOW[change?.own] ? deep('deepLikeHeld', { now: T[LIKE_NOW[change.own]] }) : '',
    change?.like && r.count === 'sampled' ? T.deepStartSaved : '',
    change?.like && (kept ?? []).includes('start_band') ? T.deepNotRemeasured : ''].filter(Boolean).join(' ');
  const notes = [];
  const none = nbInt < SOURCES_MIN;
  // Most of the RNA from neighbours looks like its own kind: said under the headline, how much looks like a different
  // kind in its tooltip (the headline counts it all, and genes can back up only the different kind's part).
  const lookAlike = none ? null : lookAlikeLine(code, r.same_kind_share);
  if (lookAlike) notes.push(lookAlike);
  if (change?.unjudged) notes.push({ text: T.deepStartUnjudged, tip: '' });
  // The basis: proseg's whole count of the cell, the genes the model lacks included (as its background in the tooltip
  // and the stage's readout count them), named by which count the answer reads; the split itself reads the model's genes
  // (L), those left out said below.
  const basis = deep(r.count === 'sampled' ? 'deepBasisShortSaved' : 'deepBasisShort', { n: count(L + r.outside) }), basisTip = basisTooltip(r);
  const after = [];
  if (L < FEW_TX) after.push({ text: T.deepFew, tip: deep('deepFewTooltip', { n: count(L) }) });
  if (code <= FLOOR_CODE) after.push({ text: T.deepFloor, tip: T.deepFloorTooltip });
  if (r.rough) after.push({ text: T.deepRough, tip: T.deepRoughTooltip });
  if (r.outside > 0) after.push({ text: deep('deepOutside', { n: count(r.outside) }), tip: T.deepOutsideTooltip });
  // From neighbours: the tagged genes (chips by their transcripts from neighbours), the look-alike lines, the source rows.
  const tagged = r.genes.filter(g => g.flag).sort((a, b) => b.nb - a.nb || (a.gene < b.gene ? -1 : 1));
  // A chip's first number is rounded down, so a tagged gene with under half its transcripts as a whole number
  // ("GENE1 1 of 3" for 1.6 of 3) would contradict its label: it stays in the label's count and in the table with its
  // tag, not as a chip.
  const chips = tagged.filter(g => Math.floor(g.nb + 1e-6) * 2 >= g.tx)
    .map(g => ({ key: g.gene, gene: g.gene, nb: g.nb, tx: g.tx, text: chipText(g.gene, g.nb, g.tx) }));
  const allNb = r.genes.reduce((s, g) => s + g.nb, 0), taggedNb = tagged.reduce((s, g) => s + g.nb, 0);
  const spread = tagged.length > 0 && allNb > 0 && taggedNb < 0.5 * allNb;
  const sameShare = r.same_kind_share;
  const allSame = !tagged.length && sameShare >= 0.75;
  const neighbours = r.neighbours.map((n, k) => {
    const lab = label(n.cell), g = lab ? group(lab) : null;
    const kindWords = genesList(n.kind.slice(0, 3), ' · ');
    const makes = kindWords ? deep('deepSourceMakes', { genes: kindWords }) : null;
    const groupText = g?.label || makes || T.deepGroupUnknown;
    const share = L > 0 ? n.tx / L : 0, pct = partPercent(share, nbInt);
    // The genes it seems to give, with their transcripts from it, whole numbers rounded down; a neighbour of the same
    // kind says that its share can't be checked instead. Its own kind's genes and its share go to its tooltip.
    const genes = n.genes.slice(0, SOURCE_GENES).filter(([, tx]) => Math.floor(tx + 1e-6) >= 1)
      .map(([gene, tx]) => deep('deepSourceGene', { gene, n: number(Math.floor(tx + 1e-6)) }));
    const distance = finite(n.um) ? deep('deepSourceDistance', { d: number(Math.round(n.um)) }) : '';
    const head = deep('deepSourceRow', { cell: name(n.cell), group: groupText });
    const tooltip = [pct < 1 ? deep('deepSourceTooltipSmall', { head }) : deep('deepSourceTooltip', { head, p: number(pct) }),
      g?.label && kindWords ? deep('deepSourceKind', { genes: kindWords }) : '', n.same ? T.deepSourceSameKindTooltip : ''].filter(Boolean).join(' ');
    const genesText = n.same ? T.deepSourceSameKind : genes.join(' · ');
    return { key: lab || `n${k}`, index: k, cell: n.cell, label: lab, name: name(n.cell), share, pct, pctText: sharePct(share, nbInt),
      tx: n.tx, um: n.um, same: n.same, known: n.known, group: g, groupText, makes, kind: n.kind, color: g ? clusterColorCSS(g.color) : null,
      head, distance, genes: genesText, sub: [genesText, distance].filter(Boolean).join(' · '), tooltip };
  });
  const candidates = neighbours.filter(n => n.share >= ROW_MIN);
  const sameCount = r.neighbours.filter(n => n.same).length;
  // Its own and loose RNA.
  const kind = genesList(r.kind.slice(0, KIND_GENES), ' · ');
  const ownGenes = r.own_genes.slice(0, OWN_GENES).map(([g, own]) => deep('deepOwnGene', { gene: g, n: number(Math.round(own)) }));
  const looseP = Math.max(0.1, Math.round(r.parts[2] / ((r.parts[0] + r.parts[1] + r.parts[2]) || 1) * 1000) / 10);
  // Every gene, by transcripts; the neighbour giving most of a gene when the worker names it (its 6th field, from the
  // whole split: a neighbour's top-4 list could name the wrong one), with the row's tooltip without it (`unnamed`: while
  // the glow lies near none of that neighbour, QcDeeper.tableTitle). Made when the table is first read (`table`): it is
  // collapsed until opened, and a cell's hundreds of rows and tooltips were most of each apply's work (P5).
  let rows = null;
  const tableRows = () => rows ??= [...r.genes].sort((a, b) => b.tx - a.tx || (a.gene < b.gene ? -1 : 1)).map(g => {
    const giver = Number.isInteger(g.giver) && r.neighbours[g.giver] ? neighbours[g.giver] : null;
    return { gene: g.gene, tx: g.tx, own: g.own, nb: g.nb, loose: g.loose, flag: g.flag,
      pct: number(g.tx > 0 ? Math.round(g.nb / g.tx * 100) : 0),
      tooltip: rowTooltip(g.gene, g.tx, g.own, g.nb, giver?.name ?? null),
      giver: giver ? { label: giver.label, name: giver.name } : null, unnamed: giver ? rowTooltip(g.gene, g.tx, g.own, g.nb) : null };
  });
  return {
    state: 'ready', code, p, number: deep('deepNumber', { p }), nb: nbInt, step, stepLine, stepTip, basis, basisTip, notes, after, bar,
    barLabel: deep('deepBarLabel', { own: bar.texts.own, nb: bar.texts.nb, loose: bar.texts.loose }),
    none, noneText: none ? T.deepSourcesNone : null,
    chips, chipsTip: tagged.length ? [deep('deepForeignHeading', { k: number(tagged.length) }), spread ? T.deepSpread : ''].filter(Boolean).join(' ') : null,
    foreignNone: !tagged.length && !allSame ? T.deepForeignNone : null,
    // From half on, the line under the headline says it (lookAlikeLine); a quarter to a half: here, short.
    sameLine: !allSame && sameShare >= 0.25 && !lookAlike ? { text: deep('deepSameKindShare', { p: number(Math.round(sameShare * 100)) }), tip: T.deepSameKindShareTooltip } : null,
    neighbours, candidates, learned: r.learned,
    // The rows' heading's tooltip: the learned guide, or the rougher one for a cell the model didn't learn.
    learnedNote: r.learned === 'no' ? T.deepNotLearnedNote : T.deepLearnedNote,
    sameCount: sameKindCountText(sameCount, r.neighbours.length),
    ownKind: kind ? deep('deepOwnKind', { genes: kind }) : null,
    ownGenes: ownGenes.length ? { text: deep('deepOwnGenes', { genes: ownGenes.join(' · ') }), tip: T.deepOwnGenesTooltip }
      : { text: T.deepOwnGenesNone, tip: T.deepOwnGenesNoneTooltip },
    // Its top genes as a sentence of the own kind line's tooltip (that line alone on screen).
    ownTop: ownGenes.length ? deep('deepOwnTop', { genes: ownGenes.join(' · ') }) : T.deepOwnTopNone,
    // Loose RNA under 1 % is the legend's "< 1 %" alone.
    loose: bar.looseUnder ? null : { text: deep('deepLooseLine', { p: number(bar.loose) }), tip: deep('deepLooseBody', { p: number(looseP, 1) }) },
    get table() { return tableRows(); }, tableHeading: deep('deepTableHeading', { k: number(r.genes_total ?? r.genes.length) }),
    tableRest: r.rest ? deep('deepTableRest', { k: number(r.rest.k), tx: count(r.rest.tx) }) : null,
  };
}
/** The basis's tooltip for result `r`: the start line's own (proseg's count at iteration 0, every gene), else what
 *  proseg's count of the cell is (averaged over its random background draw, or as proseg saves it: r.count), with what
 *  proseg reads as background in the result's own frame (`stats` {assigned, kept}: that frame's statistics, kept being
 *  the same count, assigned less kept rounded, so the two whole numbers add up to the cell's transcripts, as the stage's
 *  readout says them: cellstats.mjs readoutText) once they are in. */
export function basisTooltip(r, stats = null) {
  if (r?.seq === 'start') return deep('deepBasisStart', { n: count((r.transcripts ?? 0) + (r.outside ?? 0)) });
  const known = stats && finite(stats.assigned) && finite(stats.kept) && stats.assigned >= 0 && stats.kept >= 0, saved = r?.count === 'sampled';
  return known ? deep(saved ? 'deepBasisSaved' : 'deepBasis', { bg: count(Math.max(0, stats.assigned - Math.round(stats.kept))) })
    : saved ? T.deepBasisNoBgSaved : T.deepBasisNoBg;
}
/** Whether the window names the cell groups' iteration (deepGroupsFrom): only when they are GROUPS_LAG or more
 *  iterations from the result's, or newer than it (a scrub back); a pass or two of lag is the usual and goes to the
 *  Log. */
export function groupsLag(groupsIter, iter) {
  return finite(groupsIter) && finite(iter) && (groupsIter > iter || Math.abs(groupsIter - iter) >= GROUPS_LAG);
}
/** The rows the window lists: candidates (>= 1 % of the cell's RNA) ranked with hysteresis from `previous`, at most
 *  ROWS; the rest fold into one line. Returns {rows, fold: {k, share, text} | null, partly}. */
export function sourceRows(model, previous = []) {
  const byKey = new Map(model.candidates.map(n => [n.key, n]));
  const order = rankOrder(previous, [...byKey.keys()], k => byKey.get(k).share).slice(0, ROWS);
  const rows = order.map(k => byKey.get(k)), shown = new Set(order);
  const rest = model.neighbours.filter(n => !shown.has(n.key));
  const restShare = rest.reduce((s, n) => s + n.share, 0);
  return { rows, fold: rest.length && rows.length ? { k: rest.length, share: restShare, text: foldText(rest.length, restShare, model.nb) } : null,
    partly: model.learned === 'no' ? null : partlyText(rows.filter(r => !r.known).length) };
}

// ---- where it spills (pure: node tests) ------------------------------------------------------------------------------
/** What "Show on the image" asks for the ready result `r` shown on cell `cell` (label) with its source rows `rows` as
 *  painted (their order is the reply's source order) and the chips' ranked genes `chips`: {key, params: {seq, cell,
 *  sources, genes, own} for api.qcDeeperWhere, labels (the sources'), sources: [{name, head, same, kind (its own kind's
 *  genes)}]}; null when there is nothing to show (no ready result, under 5 % from neighbours, no source row). The exact
 *  twin of engine/qc_scoring.py where_request (the shared fixture holds its key); the window's own key adds the
 *  result's count (QcDeeper.whereWant). */
export function whereRequest(r, { cell = 0, none = false, rows = [], chips = [] } = {}) {
  if (!r || r.state !== 'ready' || none || !(cell > 0)) return null;
  const shown = rows.filter(row => row?.label > 0).slice(0, ROWS);
  if (!shown.length) return null;
  const genes = [...new Set(chips.filter(g => typeof g === 'string' && g))].slice(0, WHERE_GENES);
  const own = [...new Set((r.own_genes || []).map(g => g[0]).filter(g => typeof g === 'string' && g && !genes.includes(g)))].slice(0, WHERE_OWN);
  const params = { seq: r.seq, cell: cell - 1, sources: shown.map(row => row.label - 1), genes, own };
  return { key: JSON.stringify([params.seq, params.cell, params.sources, genes, own]), params, labels: shown.map(row => row.label),
    sources: shown.map(row => ({ name: row.name, head: row.head, same: !!row.same, kind: Array.isArray(row.kind) ? [...row.kind] : [] })) };
}
const pair = v => (Array.isArray(v) && v.length === 2 && v.every(x => finite(x) && x >= 0) ? [v[0], v[1]] : null);
const maybe = v => (finite(v) && v >= 0 ? v : null);
/** A reply of GET /runs/{id}/qc/deeper/where (v 3: points.w, proseg's share of each point for the cell, 0 to 1, and
 *  `held`, where those shares came from: "exact", "approx" or "none"; points.near: 0/1 per point, a "from" gene's point
 *  proseg counts for the cell (or inside its outline) within 2 µm of its giver, or a source's own-gene point within 2 µm
 *  of the cell) checked against its request `req` and normalised (the points as typed arrays: w as the doubles the reply
 *  gives; without near flags all 0; z, each point's depth as the engine gives it, normalised to 3 decimals: NaN where it
 *  gives none; `form`, the count its shares read: the reply's `count`, "sampled" or, absent too, "averaged"; `count`
 *  is the number of points); null when it is not one. Lenient about extra fields. */
export function normaliseWhere(raw, req = null) {
  if (!raw || typeof raw !== 'object' || !WHERE_V.includes(raw.v)) return null;
  if (req && (raw.cell !== req.params.cell || String(raw.seq) !== String(req.params.seq))) return null;
  const genes = raw.genes, roles = raw.roles;
  if (!Array.isArray(genes) || !Array.isArray(roles) || genes.length !== roles.length || !genes.every(g => typeof g === 'string')
    || !roles.every(role => WHERE_ROLES.has(role))) return null;
  const box = raw.box ?? null;   // null: neither the cell nor a source holds anything in that frame (no dots)
  if (box !== null && (!Array.isArray(box) || box.length !== 4 || !box.every(finite) || !(box[2] > box[0] && box[3] > box[1]))) return null;
  const p = raw.points && typeof raw.points === 'object' ? raw.points : null, n = Array.isArray(p?.x) ? p.x.length : -1;
  if (n < 0 || !['y', 'gene', 'owner', 'w'].every(k => Array.isArray(p[k]) && p[k].length === n)) return null;
  if (p.near != null && (!Array.isArray(p.near) || p.near.length !== n)) return null;
  const x = new Float32Array(n), y = new Float32Array(n), gene = new Uint8Array(n), owner = new Uint8Array(n), near = new Uint8Array(n);
  const z = new Float32Array(n), depth = Array.isArray(p.z) && p.z.length === n ? p.z : null, w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const g = p.gene[i], o = p.owner[i], c = p.near ? p.near[i] : 0, share = p.w[i];
    if (!finite(p.x[i]) || !finite(p.y[i]) || !Number.isInteger(g) || g < 0 || g >= genes.length || !Number.isInteger(o) || o < 0 || o > 255
      || (c !== 0 && c !== 1) || !finite(share) || share < 0 || share > 1) return null;
    x[i] = p.x[i]; y[i] = p.y[i]; gene[i] = g; owner[i] = o; near[i] = c; w[i] = share; z[i] = depth && finite(depth[i]) ? depth[i] : NaN;
  }
  const stats = (Array.isArray(raw.gene_stats) ? raw.gene_stats : []).filter(s => s && typeof s.gene === 'string').map(s => ({
    gene: s.gene, role: WHERE_ROLES.has(s.role) ? s.role : 'from', giver: Number.isInteger(s.giver) && s.giver >= 0 ? s.giver : null,
    in_cell: num(s.in_cell), near_giver: maybe(s.near_giver), own_near: pair(s.own_near),
    density: { cell: maybe(s.density?.cell), giver: maybe(s.density?.giver), around: maybe(s.density?.around) },
    count: maybe(s.count), word: Object.hasOwn(WHERE_WORDS, s.word) ? s.word : null }));
  const into = (Array.isArray(raw.into) ? raw.into : []).filter(t => t && typeof t.gene === 'string' && Number.isInteger(t.source) && t.source >= 0)
    .map(t => ({ source: t.source, gene: t.gene, inside: num(t.inside), near_cell: num(t.near_cell), source_own_near: pair(t.source_own_near) }));
  return { v: raw.v, seq: raw.seq, iter: finite(raw.iter) ? raw.iter : null, cell: raw.cell, basis: typeof raw.basis === 'string' ? raw.basis : null,
    held: HELD.includes(raw.held) ? raw.held : null, held_seq: Number.isInteger(raw.held_seq) && raw.held_seq >= 0 ? raw.held_seq : null,
    form: raw.count === 'sampled' ? 'sampled' : 'averaged',
    box: box && [...box], genes: [...genes], roles: [...roles], count: n, x, y, z, gene, owner, near, w, truncated: raw.truncated === true,
    cell_total: finite(raw.cell_total) && raw.cell_total >= 0 ? raw.cell_total : null, stats, into };
}
/** Colour slots that stay with their genes: `slots` (Map gene -> slot, kept while the toggle is on) gives every gene of
 *  `genes` that has none the lowest free slot (under `size`), in order; genes no longer asked free theirs. Returns slots. */
export function assignSlots(slots, genes, size) {
  for (const g of [...slots.keys()]) if (!genes.includes(g)) slots.delete(g);
  const used = new Set(slots.values());
  for (const g of genes) {
    if (slots.has(g)) continue;
    let k = 0; while (used.has(k)) k++;
    if (k < size) { slots.set(g, k); used.add(k); }
  }
  return slots;
}
/** The stage's dots for a reply (normaliseWhere) with `sources` source rows: x, y (µm), the gene's colour index (its place
 *  in reply.genes) and the dot's opacity (WHERE_ALPHA), four floats each, in draw order: own genes where they are
 *  expected, then the chips' genes around, then own genes inside the sources, then the chips' genes in the cell and its
 *  sources (on top). */
export function wherePoints(reply, sources = ROWS) {
  const lists = [[], [], [], []], alpha = [WHERE_ALPHA.expected, WHERE_ALPHA.around, WHERE_ALPHA.lit, WHERE_ALPHA.lit];
  for (let i = 0; i < reply.count; i++) {
    const o = reply.owner[i], inSource = o >= 2 && o < 2 + sources;
    lists[reply.roles[reply.gene[i]] === 'own' ? (inSource ? 2 : 0) : (inSource || o === OWNER_CELL ? 3 : 1)].push(i);
  }
  const out = new Float32Array(4 * reply.count);
  let j = 0;
  lists.forEach((list, b) => { for (const i of list) { out[j++] = reply.x[i]; out[j++] = reply.y[i]; out[j++] = reply.gene[i]; out[j++] = alpha[b]; } });
  return out;
}
/** The part of the stage {width, height} (CSS px) the camera's move for "Show on the image" puts `box` ([x0, y0, x1, y1]
 *  µm: the cell and its sources with their margin) in: [x0, y0, x1, y1] CSS px, WHERE_MARGIN inside the stage and clear
 *  of the window `win` ({left, top, w, h} CSS px, or null) beside it: the wider side when it holds at least 45 % of the
 *  stage, else the whole stage. The open test corner `corner` ({left, top, w, h}, or {left, bottom, w, h}; null when
 *  closed) is kept clear of in that part as well: the box goes above, below, left or right of it, wherever it shows
 *  largest, while that still shows it at least 45 % as large. whereView centres the box in it (2-D); in 3-D the stage's
 *  fit takes it with the box (Stage.fit3d). */
export function whereFree(box, stage, win = null, corner = null) {
  const W = stage?.width > 0 ? stage.width : 800, H = stage?.height > 0 ? stage.height : 600, M = WHERE_MARGIN;
  const bw = Math.max(1, box[2] - box[0]), bh = Math.max(1, box[3] - box[1]);
  let x0 = M, x1 = W - M, y0 = M, y1 = H - M;
  if (win && win.w > 0) {
    const left = win.left - 2 * M, right = W - win.left - win.w - 2 * M;
    if (Math.max(left, right) >= SIDE_MIN * W) { if (right >= left) x0 = win.left + win.w + M; else x1 = win.left - M; }
  }
  const fit = (a, b, c, d) => (c > a && d > b ? Math.min((c - a) / bw, (d - b) / bh) : 0);   // CSS px per µm
  const top = finite(corner?.top) ? corner.top : finite(corner?.bottom) && finite(corner?.h) ? H - corner.bottom - corner.h : null;
  if (corner && corner.w > 0 && corner.h > 0 && finite(corner.left) && top != null
    && corner.left - M < x1 && corner.left + corner.w + M > x0 && top - M < y1 && top + corner.h + M > y0) {
    const sides = [[x0, y0, x1, top - M], [x0, top + corner.h + M, x1, y1], [x0, y0, corner.left - M, y1], [corner.left + corner.w + M, y0, x1, y1]];
    const best = sides.reduce((a, s) => (fit(...s) > fit(...a) ? s : a));
    if (fit(...best) >= SIDE_MIN * fit(x0, y0, x1, y1)) [x0, y0, x1, y1] = best;
  }
  return [x0, y0, x1, y1];
}
/** The camera's one move for "Show on the image" in 2-D: a world rect for stage.fit(rect, animate, 0) that shows `box`
 *  ([x0, y0, x1, y1] µm) as large as it fits, centred in the part of the stage whereFree leaves for it (`stage`, `win`
 *  and `corner` as there). */
export function whereView(box, stage, win = null, corner = null) {
  const W = stage?.width > 0 ? stage.width : 800, H = stage?.height > 0 ? stage.height : 600;
  const bw = Math.max(1, box[2] - box[0]), bh = Math.max(1, box[3] - box[1]);
  const [x0, y0, x1, y1] = whereFree(box, stage, win, corner);
  const fit = (a, b, c, d) => (c > a && d > b ? Math.min((c - a) / bw, (d - b) / bh) : 0);   // CSS px per µm
  const zoom = Math.max(1e-6, fit(x0, y0, x1, y1));
  const cx = (box[0] + box[2]) / 2 - ((x0 + x1) / 2 - W / 2) / zoom, cy = (box[1] + box[3]) / 2 - ((y0 + y1) / 2 - H / 2) / zoom;
  return { x0: cx - W / 2 / zoom, y0: cy - H / 2 / zoom, w: W / zoom, h: H / zoom };
}
/** A density (transcripts per 100 µm²), or a sum of proseg's shares, as "Where they sit" says it: a whole number,
 *  "under 1" below 1, 0 for none. */
export const whereDensity = v => (!(v > 0) ? '0' : v < 1 ? T.deepWhereUnder : number(Math.round(v)));

// ---- the suspect transcripts and their glow (pure: node tests) ----------------------------------------------------------
/** Which of the transcripts proseg counts for the cell likely came from a neighbour: the model splits each gene as a
 *  whole, their places add where. For a "from" gene g of `reply` (normaliseWhere) with giver k (a source row), from the
 *  window's own result row (`result`: the normalised ready result the request was made for, never the engine's current
 *  model): s = nb / tx; the rest of the cell's RNA within 2 µm of k, smoothed, f_r = (own_near[0] + 1) / (own_near[1] +
 *  2); g's share there f_g = near_giver / in_cell (proseg's count of the cell, as the reply's sums of w); f_s =
 *  clamp((f_g - (1 - s) f_r) / s, f_r, 1), how close to k the neighbour's part of g must lie to explain f_g. A transcript
 *  of g within 2 µm of k (the reply's near flag) gets pi = s f_s / (s f_s + (1 - s) f_r), one further in pi = s (1 - f_s)
 *  / (s (1 - f_s) + (1 - s)(1 - f_r)) (0 for 0 / 0), and P = w pi, w proseg's share of it for the cell (0: no suspect,
 *  wherever it lies). Unclamped, g's P add up to s times proseg's count of g; with no spatial evidence (f_g = f_r) every
 *  one gets s. A gene with no giver among the rows: s for each of its transcripts; a gene not in the result: none.
 *  Returns {p, pi: Float32Arrays (per point; 0 for a point that is no suspect), genes: [{gene, giver, s, fr, fg, fs,
 *  near, far}]}. */
export function suspectScores(reply, result) {
  const n = reply?.count ?? 0, p = new Float32Array(n), pi = new Float32Array(n);
  if (!n || result?.state !== 'ready' || !Array.isArray(result.genes)) return { p, pi, genes: [] };
  const rows = new Map(result.genes.map(g => [g.gene, g]));
  const stats = new Map((reply.stats ?? []).filter(s => s.role === 'from').map(s => [s.gene, s]));
  const per = reply.genes.map((gene, i) => {
    const row = reply.roles[i] === 'from' ? rows.get(gene) : null;
    if (!row || !(row.tx > 0)) return null;
    const s = clamp(row.nb / row.tx, 0, 1), st = stats.get(gene), k = st?.giver ?? null;
    if (k == null || !st.own_near || st.near_giver == null || !(st.in_cell > 0)) return { gene, giver: null, s, near: s, far: s };
    const fr = (st.own_near[0] + 1) / (st.own_near[1] + 2), fg = clamp(st.near_giver / st.in_cell, 0, 1);
    if (!(s > 0)) return { gene, giver: k, s: 0, fr, fg, fs: fr, near: 0, far: 0 };
    const fs = clamp((fg - (1 - s) * fr) / s, fr, 1), farIn = s * (1 - fs) + (1 - s) * (1 - fr);
    return { gene, giver: k, s, fr, fg, fs, near: s * fs / (s * fs + (1 - s) * fr), far: farIn > 0 ? s * (1 - fs) / farIn : 0 };
  });
  for (let i = 0; i < n; i++) {
    const g = reply.w[i] > 0 ? per[reply.gene[i]] : null;
    if (!g) continue;
    const v = reply.near?.[i] ? g.near : g.far;
    pi[i] = v; p[i] = reply.w[i] * v;
  }
  return { p, pi, genes: per.filter(Boolean) };
}
/** How much of the glow lies near each source: for source k (its place in the request's sources), proseg's share
 *  summed over the glowing points (P >= GLOW_MIN) whose gene k gives and whose near flag is set (within 2 µm of k), w
 *  added up in point order. Map(k -> support), a source with none left out (0): one the glow would name with none is not
 *  where its pink dots come from. */
export function sourceSupport(reply, scores) {
  const out = new Map(), n = reply?.count ?? 0, P = scores?.p;
  if (!n || !P) return out;
  const giverOf = new Map((reply.stats ?? []).filter(s => s.role === 'from').map(s => [s.gene, s.giver]));
  const givers = reply.genes.map(g => giverOf.get(g) ?? null);
  for (let i = 0; i < n; i++) {
    if (!(P[i] >= GLOW_MIN) || !reply.near[i]) continue;
    const k = givers[reply.gene[i]];
    if (k != null) out.set(k, (out.get(k) ?? 0) + reply.w[i]);
  }
  return out;
}
/** Where each transcript of `reply` proseg counts for the cell (w > 0, the ones that can glow, inside its outline or
 *  moved in) sat at the start, from `start`, the start line's reply (normaliseWhere) to the same cell and genes: its
 *  point of the same gene and place there (x and y within MATCH_UM: the replies give them to 0.01 µm; where two such
 *  points lie in different layers, the one of the nearest depth) and that point's owner then, as a code of PLACES
 *  (Uint8Array(reply.count); 0 for a point proseg does not count for the cell):
 *  - 1 own: the cell's (owner 1), inside its own starting outline;
 *  - 2 none: no cell's (owner 0: no starting cell there, or outside the start's grid);
 *  - 3 giver: the neighbour its gene seems to come from (the gene's giver in the reply: source k of `sources`, the
 *    frame request's; the start request lists the same cells sorted, `startSources`, as whereStart asks it);
 *  - 4 other: any other cell's (another source, or 255);
 *  - 5 unknown: no such point in the start reply (it draws only a box around the cell and its neighbours at the start,
 *    a long reply is cut short, and it is not asked for the cell's own genes, which never glow).
 *  The engine places the test's suspects by the same classes (on the start's own layers). Null without a start reply
 *  for that cell holding every "from" gene of the reply, or one that lists another set of neighbours. The start reply's
 *  index is built once per reply (startIndex), so a live view pays only for each frame's own points. */
export function startPlaces(reply, start, sources = [], startSources = [...sources].sort((a, b) => a - b)) {
  if (!reply || !start || start.cell !== reply.cell) return null;
  const from = reply.genes.filter((g, i) => reply.roles[i] === 'from');
  if (!from.every(g => start.genes.includes(g)) || sources.length !== startSources.length || !sources.every(s => startSources.includes(s))) return null;
  // Each gene's giver as the start reply's owner code: 2 + its place in the start's sources (none: -1).
  const giverOf = new Map((reply.stats ?? []).filter(s => s.role === 'from').map(s => [s.gene, s.giver]));
  const giverCode = reply.genes.map(g => { const k = giverOf.get(g); return Number.isInteger(k) && k < sources.length ? 2 + startSources.indexOf(sources[k]) : -1; });
  const grid = startIndex(start);
  const out = new Uint8Array(reply.count), tol = MATCH_UM + 1e-3;   // float32 places: within a thousandth of a µm
  for (let i = 0; i < reply.count; i++) {
    if (!(reply.w?.[i] > 0)) continue;
    const gene = reply.genes[reply.gene[i]], gx = onGrid(reply.x[i]), gy = onGrid(reply.y[i]), z = reply.z?.[i];
    let best = -1, depth = Infinity;
    for (let d = 0; d < 9; d++) {
      for (const j of grid.get(placeKey(gene, gx + (d % 3) - 1, gy + Math.floor(d / 3) - 1)) ?? []) {
        if (Math.abs(start.x[j] - reply.x[i]) > tol || Math.abs(start.y[j] - reply.y[i]) > tol) continue;
        const dz = finite(z) && finite(start.z?.[j]) ? Math.abs(start.z[j] - z) : Infinity;
        if (best < 0 || dz < depth) { best = j; depth = dz; }
      }
    }
    if (best < 0) { out[i] = 5; continue; }
    const o = start.owner[best];
    out[i] = o === OWNER_CELL ? 1 : o === 0 ? 2 : o === giverCode[reply.gene[i]] ? 3 : 4;
  }
  return out;
}
// A place on the MATCH_UM grid (the replies give places to 0.01 µm), and a point's key there: its gene and that place.
const onGrid = v => Math.round(v / MATCH_UM), placeKey = (gene, x, y) => `${gene}|${x}|${y}`;
const START_INDEX = new WeakMap();
/** The start line's reply (normaliseWhere's, never changed once made) indexed for startPlaces: Map(gene and place on
 *  the MATCH_UM grid -> its points' indices), any owner. Built once per reply and kept with it (a weak key: it goes with
 *  the reply), since a live view places every frame's reply against the same start reply: about 11 ms at a reply's
 *  30,000 points, now paid once. */
export function startIndex(start) {
  let grid = START_INDEX.get(start);
  if (grid) return grid;
  grid = new Map();
  for (let j = 0; j < start.count; j++) {
    const k = placeKey(start.genes[start.gene[j]], onGrid(start.x[j]), onGrid(start.y[j]));
    const list = grid.get(k); if (list) list.push(j); else grid.set(k, [j]);
  }
  START_INDEX.set(start, grid);
  return grid;
}
/** Which of the reply's points `indices` share their place with others (two
 *  transcripts of a gene at one x and y in two layers can be two cells', and a test matches a suspect to its run's
 *  transcripts by gene and place): for point i, the reply's points of its gene whose x and y each lie within MATCH_UM of
 *  its own, any owner, itself included (the replies give places to 0.01 µm, so on that grid at most a step away on each
 *  axis). Where they are more than one: {rank: its place among them by depth, z ascending as the reply gives it (to 3
 *  decimals; equal depths in the reply's order, the run's record order: as the engine ranks them, engine/qc_test.py
 *  match), of: how many}; left out where one of them has no depth (which is which can't be said). Returns Map(i ->
 *  {rank, of}). Only the points of their genes are indexed, by their places as numbers (a reply holds up to 30,000
 *  points, applied within a few ms). */
export function samePlace(reply, indices) {
  const out = new Map(), n = reply?.count ?? 0, z = reply?.z;
  if (!n || !indices?.length) return out;
  // A place's number: its steps of MATCH_UM in x and y, each within 2^25 of the origin (335 mm), as one integer.
  const SPAN = 2 ** 26, at = (x, y) => (Math.round(x / MATCH_UM) + SPAN / 2) * SPAN + Math.round(y / MATCH_UM) + SPAN / 2;
  const grids = new Map();
  for (const i of indices) if (!grids.has(reply.gene[i])) grids.set(reply.gene[i], new Map());
  for (let j = 0; j < n; j++) {
    const grid = grids.get(reply.gene[j]);
    if (!grid) continue;
    const k = at(reply.x[j], reply.y[j]), list = grid.get(k);
    if (list) list.push(j); else grid.set(k, [j]);
  }
  for (const i of indices) {
    const grid = grids.get(reply.gene[i]), k = at(reply.x[i], reply.y[i]), group = [];
    for (let d = 0; d < 9; d++) for (const j of grid.get(k + ((d % 3) - 1) * SPAN + Math.floor(d / 3) - 1) ?? []) group.push(j);
    if (group.length < 2 || !z || !group.every(j => finite(z[j]))) continue;
    group.sort((a, b) => z[a] - z[b] || a - b);
    out.set(i, { rank: group.indexOf(i), of: group.length });
  }
  return out;
}
/** The glow for the stage: the suspects at P >= GLOW_MIN of every giver (k null) or of source row k
 *  only (its place in the request's sources), whoever's outline they lie in, as [x, y, P] triples, dimmest first (the
 *  brightest drawn last), and `depth` (Float32Array, in the same order: each one's z as the reply gives it, NaN for
 *  none; the glow stands there in 3-D); `count` of them, `likely` (pi >= LIKELY_MIN where proseg counts at least LIKELY_W of it for
 *  the cell) and `near` (likely ones within 2 µm of their giver); `places`: of those the line counts, the likely ones or,
 *  with none, every glowing one, how many sat where at the start ({own, none, giver, other, unknown} from startPlaces'
 *  codes `places`; null without them), and `start` its `own` (null without them); `levels`: of those inside its own
 *  starting outline, how the start held them ({nucleus, cell, unknown} from the start's hold `holds`, HOLD codes per
 *  point; null without places or holds); `giver`: k, or for every giver the one whose suspects weigh most, the glow's
 *  side to ripple in from; and `notNear`: that source when none of the glowing points lies near it (sourceSupport: it is
 *  not where they come from), `giver` then null (no side to ripple in from; null for no glow either). */
export function glowPoints(reply, scores, k = null, places = null, holds = null) {
  const n = reply?.count ?? 0, P = scores?.p, pi = scores?.pi, idx = [], weight = new Map();
  const giverOf = new Map((reply?.stats ?? []).filter(s => s.role === 'from').map(s => [s.gene, s.giver]));
  const givers = (reply?.genes ?? []).map(g => giverOf.get(g) ?? null);
  for (let i = 0; i < n && P; i++) {
    if (!(P[i] >= GLOW_MIN)) continue;
    const g = givers[reply.gene[i]];
    if (k != null && g !== k) continue;
    idx.push(i);
    if (g != null) weight.set(g, (weight.get(g) ?? 0) + P[i]);
  }
  idx.sort((a, b) => P[a] - P[b] || a - b);
  const points = new Float32Array(3 * idx.length), depth = new Float32Array(idx.length);
  // Per place (and per level of those inside its own outline): every glowing one [0] and the likely ones [1].
  const tally = () => Object.fromEntries(PLACES.map(p => [p, 0])), level = () => ({ nucleus: 0, cell: 0, unknown: 0 });
  const where = places ? [tally(), tally()] : null, held = places && holds ? [level(), level()] : null;
  let likely = 0, near = 0;
  idx.forEach((i, j) => {
    points[3 * j] = reply.x[i]; points[3 * j + 1] = reply.y[i]; points[3 * j + 2] = P[i]; depth[j] = finite(reply.z?.[i]) ? reply.z[i] : NaN;
    const sure = pi?.[i] >= LIKELY_MIN && reply.w?.[i] >= LIKELY_W, place = places ? PLACES[places[i] - 1] : undefined;
    if (sure) { likely++; if (reply.near[i]) near++; }
    if (!place) return;
    for (const t of sure ? [0, 1] : [0]) {
      where[t][place]++;
      if (held && place === 'own') held[t][HOLD_NAMES[holds[i]] ?? 'unknown']++;
    }
  });
  let giver = k;
  if (k == null) for (const [g, w] of weight) if (giver == null || w > weight.get(giver)) giver = g;
  const notNear = idx.length && giver != null && !((sourceSupport(reply, scores).get(giver) ?? 0) > 0) ? giver : null;
  const counted = likely ? 1 : 0;
  return { points, depth, count: idx.length, likely, near, start: where ? where[counted].own : null, places: where ? where[counted] : null,
    levels: held ? held[counted] : null, giver: notNear != null ? null : giver, notNear };
}
/** Where source k (its place in the request's sources) lies as the reply shows it: the mean place of its transcripts
 *  there ([x, y] µm, the glow's side to ripple in from), or null when it holds none of them. */
export function sourceCentre(reply, k) {
  let sx = 0, sy = 0, n = 0;
  for (let i = 0; i < (reply?.count ?? 0); i++) if (reply.owner[i] === 2 + k) { sx += reply.x[i]; sy += reply.y[i]; n++; }
  return n ? [sx / n, sy / n] : null;
}
/** The box [x0, y0, x1, y1] µm around the cell (the transcripts inside its outline) and the glowing points (`points`:
 *  glowPoints' [x, y, P] triples), PAIR_MARGIN_UM wider on every side: the camera's box when none of the glow lies near
 *  the neighbour it would name (that neighbour may lie far off); null when there is neither. */
export function glowBoxOf(reply, points) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, found = 0;
  const take = (x, y) => { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); found++; };
  for (let i = 0; i < (reply?.count ?? 0); i++) if (reply.owner[i] === OWNER_CELL) take(reply.x[i], reply.y[i]);
  for (let j = 0; j + 2 < (points?.length ?? 0); j += 3) take(points[j], points[j + 1]);
  const M = PAIR_MARGIN_UM;
  return found ? [x0 - M, y0 - M, x1 + M, y1 + M] : null;
}
/** The box [x0, y0, x1, y1] µm around the cell and source k as the reply shows them (their transcripts), PAIR_MARGIN_UM
 *  wider on every side: a row's Show fits the camera to it; null when either holds none. */
export function pairBox(reply, k) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, cell = 0, source = 0;
  for (let i = 0; i < (reply?.count ?? 0); i++) {
    const o = reply.owner[i];
    if (o === OWNER_CELL) cell++; else if (o === 2 + k) source++; else continue;
    x0 = Math.min(x0, reply.x[i]); y0 = Math.min(y0, reply.y[i]); x1 = Math.max(x1, reply.x[i]); y1 = Math.max(y1, reply.y[i]);
  }
  const M = PAIR_MARGIN_UM;
  return cell && source ? [x0 - M, y0 - M, x1 + M, y1 + M] : null;
}
/** "Where they sit" for a reply (normaliseWhere) to request `req` (whereRequest), with `colors[i]` the colour of reply gene
 *  i and `glow` the glow shown ({count, likely, near, start, places, levels, row: its row's label or null, notNear}:
 *  glowPoints', or null): per source (in the rows' order) its name and head, the chips' genes that seem to come from it
 *  (by proseg's count of them in this cell) with their word, their numbers in the line's tooltip, and, with proseg's
 *  exact shares (held "exact": approximate ones put none of a cell's share inside a neighbour's outline, by how they are
 *  made), one line of how many of this cell's own genes inside its outline proseg counts for this cell (each gene of 1
 *  or more; their numbers, and what it counts of the other transcripts there, in its tooltip); the
 *  chips' genes whose neighbour is not listed (other); the glow's line ({text: startLine's, tip: startTip's, counts:
 *  {likely, own, none, giver, other, unknown}, the places null until the start line's reply is in}: how many likely come
 *  from every neighbour or the row whose Show is on, and where those it counts sat at the start; noOutlineLine's instead
 *  when none of the glow lies near the neighbour it would name, whose genes then stand under those words, never under
 *  its name (`away`), with no word or number about it (its edge, its density); the tooltip says too when proseg's
 *  shares are approximate here; when this iteration has none, a neutral line with no verdict, the why in its tooltip)
 *  and, when some RNA from neighbours cannot glow, one line saying so (unlit); the truncation note. Whole numbers, "x of
 *  y". "Too few to tell" says proseg's own count of the gene (its `count`, from the frame's counts), and, for a gene it
 *  counts WHERE_FEW or more of, that too few of them are placed here; approximate shares (the frame's layers: they miss
 *  RNA proseg moved in) say so in every gene's tooltip. With no share held here (held
 *  "none": a tiled frame's view window misses the cell, or the frame's counts are gone) the reply's sums are of nothing:
 *  each gene stands alone, no word and no number (either would read as proseg's count), the why in its tooltip, and no
 *  unlit line. `ended`: the run ended (RUN_ENDED), so those tooltips never say to pause it. A reply read on the count
 *  proseg saves (form "sampled") ends the glow's, the genes' and the into lines' tooltips saying so, after the caveat;
 *  `kept` (the run's constants kept from other runs' measurement: "glow", "where_words") adds, before it, that the glow's
 *  or the words' thresholds are not measured again for such runs. With nothing held, neither: nothing was read.
 *  `outline`: the stage draws the cell's starting outline dashed (2-D, a run from cells), which the glow's tooltip then
 *  names (glowTip). */
export function whereModel(reply, req, colors = [], glow = null, ended = false, kept = [], outline = false) {
  const colour = g => colors[reply.genes.indexOf(g)] ?? COLORS.same;
  const held = reply.held !== 'none';
  const caveat = reply.held === 'approx' ? (ended ? T.deepGlowApproxEnded : T.deepGlowApprox)
    : reply.held === 'none' ? (ended ? T.deepGlowNoneEnded : T.deepGlowNone) : null;
  const saved = held && reply.form === 'sampled' ? T.deepWhereSaved : null;
  const remeasured = name => (held && (kept ?? []).includes(name) ? T.deepNotRemeasured : null);
  const list = req?.sources ?? [];
  // The neighbour none of the glow lies near (A2): not where its pink dots come from.
  const far = glow?.count > 0 && glow.notNear != null && list[glow.notNear] ? glow.notNear : null;
  const line = (s, away = false) => {
    if (!held) return { gene: s.gene, color: colour(s.gene), word: null, text: s.gene, tooltip: caveat, numbers: [] };
    // Under the far neighbour's no-outline words: nothing measured against it (its edge, its density) is said.
    const said = away && (s.word === 'edge' || s.word === 'spread') ? null : s.word;
    // Too few placed (approximate shares): proseg counts enough of it, too few of them are placed here to tell.
    const fewPlaced = said === 'few' && reply.held === 'approx' && s.count != null && s.count >= WHERE_FEW;
    const [word, tip] = fewPlaced ? ['deepWhereFew', null] : WHERE_WORDS[said] ?? [null, null], numbers = [];
    if (fewPlaced) numbers.push(deep('deepWhereFewPlaced', { n: count(s.count) }));
    else if (said === 'few') numbers.push(deep('deepWhereFewIn', { n: underFloor(s.count ?? s.in_cell, WHERE_FEW) }));   // under 4: never 4
    else {
      if (!away && s.giver != null && s.near_giver != null) numbers.push(s.own_near?.[1] > 0   // [0, 0]: no own genes asked
        ? deep('deepWhereNear', { n: count(s.near_giver), of: count(s.in_cell), m: count(s.own_near[0]), mof: count(s.own_near[1]) })
        : deep('deepWhereNearAlone', { n: count(s.near_giver), of: count(s.in_cell) }));
      const d = s.density;
      if (d.cell != null && d.around != null) numbers.push(!away && s.giver != null && d.giver != null
        ? deep('deepWhereDensity', { here: whereDensity(d.cell), giver: whereDensity(d.giver), around: whereDensity(d.around) })
        : deep('deepWhereDensityAround', { here: whereDensity(d.cell), around: whereDensity(d.around) }));
    }
    return { gene: s.gene, color: colour(s.gene), word: said, text: word ? deep('deepWhereGene', { gene: s.gene, word: T[word] }) : s.gene,
      tooltip: [tip ? T[tip] : '', ...numbers, caveat, word ? remeasured('where_words') : null, saved].filter(Boolean).join(' '), numbers };
  };
  const from = reply.stats.filter(s => s.role === 'from').sort((a, b) => b.in_cell - a.in_cell || (a.gene < b.gene ? -1 : 1));
  const sources = list.map((src, k) => {
    // Proseg's count, for this cell, of its own genes inside the neighbour's outline (sums of its shares): exact shares
    // only, a gene once proseg counts 1 or more of it there.
    const into = reply.held !== 'exact' ? [] : reply.into.filter(t => t.source === k && t.inside >= 1).sort((a, b) => b.inside - a.inside || (a.gene < b.gene ? -1 : 1));
    const base = into.find(t => t.source_own_near)?.source_own_near ?? null;
    return { name: src.name, head: src.head, genes: k === far ? [] : from.filter(s => s.giver === k).map(s => line(s)),
      into: into.length ? { genes: into.map(t => t.gene), color: colour(into[0].gene),
        text: deep('deepWhereIntoLine', { genes: into.map(t => deep('deepOwnGene', { gene: t.gene, n: count(t.inside) })).join(' · ') }),
        tooltip: [T.deepWhereIntoTooltip, ...into.map(t => deep('deepWhereInto', { gene: t.gene, n: count(t.inside), near: whereDensity(t.near_cell) })),
          ...(base ? [deep('deepWhereIntoBase', { n: whereDensity(base[0]), of: whereDensity(base[1]) })] : []), saved].filter(Boolean).join(' ') } : null };
  }).filter(s => s.genes.length || s.into);
  const awayGenes = far == null ? [] : from.filter(s => s.giver === far).map(s => line(s, true));
  if (awayGenes.length) sources.push({ name: noOutlineHead(list[far]), head: deep('deepSourceNoOutlineTooltip', { cell: list[far].name }),
    genes: awayGenes, into: null, away: true });
  // The line: who the glowing transcripts likely come from (every neighbour, or the row whose Show is on), then where
  // those it counts (the likely ones, else every glowing one) sat at the start, once the start line's reply is in.
  const k = glow?.row ? (req?.labels ?? []).indexOf(glow.row) : -1;
  const who = k >= 0 ? list[k]?.name ?? T.deepGlowAll : T.deepGlowAll;
  const places = glow?.count > 0 ? glow.places ?? null : null;
  const counts = glow ? { likely: glow.likely ?? 0, ...Object.fromEntries(PLACES.map(p => [p, places ? places[p] ?? 0 : null])) } : null;
  const tipOf = tip => [tip, caveat, remeasured('glow'), saved].filter(Boolean).join(' ');
  const away = far != null ? noOutlineLine({ ...glow, places }, list[far], outline) : null;
  // Without proseg's shares here (held "none") nothing was measured: no verdict, a neutral line and the why in its tooltip.
  const startAt = !glow ? null : reply.held === 'none' ? { text: T.deepGlowNoneShort, tip: tipOf(glowTip(outline)), counts }
    : away ? { text: away.text, tip: tipOf(away.tip), counts }
      : { text: startLine({ ...glow, places }, who), tip: tipOf(startTip({ ...glow, places }, outline)), counts };
  const unplaced = from.filter(s => (s.giver == null || s.giver >= list.length) && s.in_cell > 0).map(s => s.gene);
  const unlit = !glow || k >= 0 || !held ? null : list.some(s => s.same) ? { text: T.deepGlowSameKind, tip: T.deepGlowSameKindTooltip }
    : unplaced.length ? { text: deep('deepGlowUnplaced', { genes: unplaced.slice(0, 3).join(' · ') }), tip: T.deepGlowUnplacedTooltip } : null;
  return { sources, other: from.filter(s => s.giver == null || s.giver >= list.length).map(line), line: startAt, unlit,
    truncated: reply.truncated ? deep('deepWhereTruncated', { n: count(reply.count) }) : null, none: reply.count === 0 };
}

// ---- the window -------------------------------------------------------------------------------------------------------
/**
 * new QcDeeper(host, options): the window inside `host` (the stage's wrapper), hidden until open(). Options:
 * api (api.js: qcDeeper, qcDeeperStop, qcDeeperHistory, learnSampleDetail), batch (the page's PaintBatch), harness,
 * context() -> {run (its constants.kept: the constants kept from other runs' measurement, said in the tooltips that
 * use them), frames, seq, live, selected, cells, qc (qc-client.mjs view), startSource, startKind (how the run's
 * cells started, GET /runs/{id}/qc start_kind: "cells" has a start line, "paired" and "nuclei" none; null until known),
 * buttons ({start, resume, step, pause}: which of the page's run buttons work now, main.js runButtons), calm, view3d,
 * stage: {width, height}, counted(seq) -> bool (the frame has proseg's count of its own, as the page knows), sampled(seq)
 * -> bool (it reads the count proseg saves there: counted for a pause, the hold, a finish or the end), prosegFix
 * and notebook (how to update an old proseg-live, and in which session: qc-copy.mjs unavailableTooltip),
 * cellStats(label, seq) -> {assigned, kept} | null (proseg's own count of the cell in counted frame `seq`, from its
 * counted file: what it places in the cell and keeps of it, the basis's tooltip's background being the difference),
 * cornerBox() -> {left, top, w, h} | null (the open test corner: the camera's move keeps clear of
 * it), testable (bool: the page can test a setting on this cell, the mixing check's testable(): a test is wired, the
 * run is a proseg run and a cell is selected, whatever the run's or the check's state, whose reasons its form lists),
 * startHold(points, label) -> Uint8Array | null (how the start held each of `points`, {x, y, z} as a where reply has
 * them, for cell `label`: HOLD codes, 1 nucleus level, 2 cell level, 0 not known; null where the page has no start
 * layers: PriorLayers.startHold, a detail run's)}, name(label) -> the cell's name, group(label) -> {label, color, iter} |
 * null, startCode(label) -> the cell's byte at the start or null, swatch(label) -> CSS colour, and callbacks
 * onFocus(focus | null) (the stage's outlines and greying out of the cells not involved, the overlay's arrows, the UMAP's
 * rings; focus.apply: it comes with a result painted, whose arrows may wait for the page's next paint batch, while a row
 * hovered or pinned is drawn at once), onSelect(label), onFind(label), onSwitchOn(), onRetryLearning(), onChange()
 * (open, closed or another cell: the mixing check's button), onBox(box | null) (the Mixing legend keeps clear of it),
 * onSpill(spill | null) ("Show on the image": {key, cell, lit: [labels], points, colors} for Stage.setSpill),
 * onGlow(glow | null) (the suspects' glow: {key,
 * points: Float32Array of [x, y, P], depth: Float32Array of their z, giver: [x, y] | null} for Stage.setGlow, in 2-D and
 * 3-D), onStartOutline({cell: label, name} | null) (while the view is on: the cell's starting outline, dashed white),
 * onFly(rect, animate, view) (its camera move: stage.fit(rect, animate, 0, view); `view` {box: [x0, y0, x1, y1] µm,
 * free: [x0, y0, x1, y1] CSS px} in 3-D, where the angle stays the user's, else null), onTest(label, name) ("Test a
 * setting here…" beside Show on the image, or beside the
 * check's switch while it is off: the mixing check's form for this cell), log(text). api also takes qcDeeperWhere.
 * Testing: clock {now, setTimeout, clearTimeout, requestAnimationFrame, cancelAnimationFrame}, storage, reducedMotion()
 * -> bool.
 */
export class QcDeeper {
  constructor(host, options = {}) {
    this.host = host; this.o = options; this.doc = host?.ownerDocument ?? globalThis.document;
    this.api = options.api ?? {}; this.batch = options.batch ?? null; this.harness = options.harness ?? null;
    this.clock = options.clock ?? {
      now: () => globalThis.performance?.now() ?? Date.now(),
      setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms), clearTimeout: id => globalThis.clearTimeout(id),
      requestAnimationFrame: fn => globalThis.requestAnimationFrame(fn), cancelAnimationFrame: id => globalThis.cancelAnimationFrame(id),
    };
    try { this.storage = options.storage === undefined ? globalThis.localStorage : options.storage; } catch { this.storage = null; }
    this.listeners = []; this.timers = new Map();
    this.isOpen = false; this.small = false; this.cell = 0; this.back = 0; this.selected = 0; this.left = null; this.top = null;
    this.size = { w: WIDTH, h: 320 }; this.stageNote = ''; this.startOutlineKey = null;
    this.covered = false; this.coveredMissed = false;   // Compare's full-screen view over the page (setCovered)
    this.build();
    this.reset(null);
    for (const name of COUNTERS) this.harness?.count(name, 0);
  }

  // ---- helpers ------------------------------------------------------------------------------------------------------
  el(tag, className = '', text = '') { const n = this.doc.createElement(tag); if (className) n.className = className; if (text) n.textContent = text; return n; }
  listen(node, type, fn, opts) { node?.addEventListener?.(type, fn, opts); this.listeners.push([node, type, fn, opts]); }
  ctx() { try { return this.o.context?.() ?? {}; } catch { return {}; } }
  call(name, ...args) { try { return this.o[name]?.(...args); } catch (error) { this.log(`${name} failed: ${error?.message || error}`); return undefined; } }
  log(text) { try { this.o.log?.(text); } catch { /* the log is an aid only */ } }
  later(key, ms, fn) { this.cancel(key); this.timers.set(key, this.clock.setTimeout(() => { this.timers.delete(key); fn(); }, Math.max(0, ms))); }
  cancel(key) { if (this.timers.has(key)) { this.clock.clearTimeout(this.timers.get(key)); this.timers.delete(key); } }
  now() { return this.clock.now(); }
  get reducedMotion() {
    if (this.ctx().calm) return true;
    if (this.o.reducedMotion) { try { return !!this.o.reducedMotion(); } catch { return false; } }
    try { return !!(this.media ??= globalThis.matchMedia?.('(prefers-reduced-motion: reduce)'))?.matches; } catch { return false; }
  }
  nameOf(label) { return label > 0 ? (this.o.name?.(label) ?? `cell ${label - 1}`) : ''; }
  /** An inline icon (ICONS), hidden from screen readers: the control or label that holds it is named. */
  icon(name) {
    const node = this.el('span', 'qc-deeper-icon'); node.setAttribute('aria-hidden', 'true'); node.dataset.icon = name; node.innerHTML = svg(name);
    return node;
  }
  setIcon(node, name) { if (node && node.dataset.icon !== name) { node.dataset.icon = name; node.innerHTML = svg(name); } }
  tool(action, icon, label) {
    const b = this.el('button', 'qc-deeper-tool'); b.type = 'button'; b.dataset.action = action; b.title = label; b.setAttribute('aria-label', label);
    b._icon = this.icon(icon); b.append(b._icon);
    return b;
  }
  button(text, action, fn, className = 'qc-button') {
    const b = this.el('button', className, text); b.type = 'button'; b.dataset.action = action;
    b.onclick = event => { event?.stopPropagation?.(); if (!b.disabled) fn(); };
    return b;
  }
  /** A section's label in small caps; with a tooltip, an info icon beside it says there is more (the label holds the
   *  tooltip, so the words and the icon show it alike). */
  heading(text, tip = '', className = '') {
    const h = this.el('h3', `qc-deeper-label${className ? ` ${className}` : ''}`); h.append(this.el('span', '', text));
    if (tip != null) { h.title = tip; h.append(this.icon('info')); }
    return h;
  }

  // ---- the DOM, built once ------------------------------------------------------------------------------------------
  build() {
    const el = (t, c, x) => this.el(t, c, x);
    const root = this.root = el('section', 'qc-deeper');
    root.id = 'qc-deeper'; root.hidden = true; root.tabIndex = -1;
    root.setAttribute('role', 'dialog'); root.setAttribute('aria-modal', 'false'); root.setAttribute('aria-labelledby', 'qc-deeper-title');
    root.dataset.state = 'closed'; root.dataset.cell = '';
    // Header (the drag handle): swatch, the cell (its tooltip says what the window shows), the frame tag; Find, Make
    // smaller, Close.
    const head = this.head = el('header', 'qc-deeper-head'); head.title = T.deepDrag;
    this.swatch = el('i', 'qc-deeper-swatch');
    const names = el('div', 'qc-deeper-names');
    this.title = el('h2', 'qc-deeper-cell'); this.title.id = 'qc-deeper-title'; this.title.title = T.deepSubtitle;
    this.frameNode = el('span', 'qc-deeper-frame');
    names.append(this.title, this.frameNode);
    this.findButton = this.tool('find', 'find', T.deepFind); this.smallButton = this.tool('smaller', 'minimise', T.deepSmaller); this.closeButton = this.tool('close', 'close', T.deepClose);
    const tools = el('div', 'qc-deeper-tools'); tools.append(this.findButton, this.smallButton, this.closeButton);
    head.append(this.swatch, names, tools);
    this.pinned = el('p', 'qc-deeper-pinned', T.deepPinned); this.pinned.title = T.deepPinnedTooltip; this.pinned.hidden = true;
    this.backButton = this.button('', 'back', () => this.goBack(), 'qc-deeper-back qc-text-button'); this.backButton.hidden = true;
    // Non-result states (off, learning, details, too few, errors) and the "Working out…" line.
    this.stateBox = el('div', 'qc-deeper-state'); this.stateBox.setAttribute('role', 'status'); this.stateBox.hidden = true;
    // The check off: its switch, and beside it "Test a setting here…" while the page can test this cell (paintWhereRow
    // moves it here; a press of its Run switches the check on first). Built once, put in the box by paintStatus.
    this.offRow = el('div', 'qc-deeper-off-row');
    this.offRow.append(this.button(T.deepSwitchOn, 'switch-on', () => { this.log('switch on asked from the window'); this.call('onSwitchOn'); }));
    // The result: headline, bar and legend (pinned), then the sections (scrolling inside).
    const body = this.body = el('div', 'qc-deeper-body'); body.hidden = true;
    const headline = el('div', 'qc-deeper-headline');
    const line = el('div', 'qc-deeper-number-line');
    this.numberNode = el('strong', 'qc-deeper-number');
    this.numberLabel = el('span', 'qc-deeper-number-label'); this.numberLabel.title = T.deepNumberTooltip;
    this.numberLabel.append(el('span', '', T.deepNumberLabel), this.icon('info'));
    // The small line and its start, the basis beside them.
    const trend = el('div', 'qc-deeper-trend');
    this.spark = el('figure', 'qc-deeper-spark'); this.sparkCanvas = el('canvas', 'qc-deeper-spark-canvas');
    this.sparkCanvas.width = 96; this.sparkCanvas.height = 28; this.sparkCanvas.setAttribute('role', 'img'); this.sparkCanvas.setAttribute('aria-label', T.deepSparkLabel);
    this.sparkCanvas.title = T.deepSparkLabel; this.sparkCaption = el('figcaption', 'qc-deeper-spark-start');
    this.spark.append(this.sparkCanvas, this.sparkCaption); this.spark.hidden = true;
    this.basisNode = el('span', 'qc-deeper-basis');
    trend.append(this.spark, this.basisNode);
    line.append(this.numberNode, this.numberLabel, trend);
    this.stepNode = el('p', 'qc-deeper-step'); this.stepNode.title = T.deepStepsTooltip;
    this.notesNode = el('div', 'qc-deeper-notes'); this.afterNode = el('div', 'qc-deeper-notes');
    headline.append(line, this.stepNode, this.notesNode, this.afterNode);
    this.bar = el('div', 'qc-deeper-bar'); this.bar.setAttribute('role', 'img');
    this.segments = ['own', 'nb', 'loose'].map(k => { const s = el('span', `qc-deeper-seg qc-deeper-part-${k}`); this.bar.append(s); return s; });
    this.legend = el('div', 'qc-deeper-legend');
    this.legendItems = [['own', T.deepOwnTooltip], ['nb', T.deepNeighboursTooltip], ['loose', T.deepLooseTooltip]].map(([k, tip]) => {
      const item = el('span', `qc-deeper-key qc-deeper-key-${k}`), sw = el('i'), text = el('span');
      item.title = tip; item.append(sw, text); this.legend.append(item); return text;
    });
    const scroll = this.scroll = el('div', 'qc-deeper-scroll');
    // From neighbours: the neighbour genes (chips), the neighbours (rows), Show on the image.
    const from = this.fromSection = el('section', 'qc-deeper-from');
    this.noneNode = el('p', 'qc-deeper-none qc-deeper-muted');
    this.foreignHeading = this.heading(T.deepForeignLabel, '', 'qc-deeper-foreign-heading'); this.chips = el('div', 'qc-deeper-foreign');
    this.moreChip = this.button('', 'more', () => { this.chipsAll = true; this.rerender(); }, 'qc-deeper-chip qc-deeper-more');
    this.foreignNone = el('p', 'qc-deeper-muted'); this.sameLine = el('p', 'qc-deeper-muted qc-deeper-same');
    this.sourcesBlock = el('div', 'qc-deeper-sources-block');
    this.sourcesHeading = this.heading(T.deepSourcesHeading, T.deepSourcesGuide, 'qc-deeper-sources-heading');
    this.stageNoteNode = el('p', 'qc-deeper-muted qc-deeper-stage-note'); this.stageNoteNode.hidden = true;
    this.sources = el('ol', 'qc-deeper-sources'); this.spreadSources = el('p', 'qc-deeper-muted', T.deepSourcesSpread);
    // A same-kind row's Show explains here (never greyed out; the rows keep their height).
    this.rowNote = el('p', 'qc-deeper-muted qc-deeper-row-note', T.deepShowSameKind); this.rowNote.setAttribute('role', 'status'); this.rowNote.hidden = true;
    this.foldNode = el('p', 'qc-deeper-fold'); this.groupsFrom = el('p', 'qc-deeper-muted');
    this.sourcesBlock.append(this.sourcesHeading, this.stageNoteNode, this.sources, this.rowNote, this.spreadSources, this.foldNode, this.groupsFrom);
    // Show on the image: a toggle under the rows (never greyed out: shown only when there is something to show), and
    // beside it in one row "Test a setting here…" (the check's own words), while it shows and the page can test this
    // cell (context testable); the toggle alone takes the row. With the check off the entry moves to the off row.
    this.whereButton = this.button('', 'where', () => this.toggleWhere(), 'qc-button qc-deeper-where-toggle');
    this.whereButton.append(this.icon('show'), el('span', '', T.deepWhereShow));
    this.whereButton.title = T.deepWhereShowTooltip; this.whereButton.setAttribute('aria-pressed', 'false'); this.whereButton.hidden = true;
    this.testButton = this.button(QC_TEXT.testOpen, 'test-open', () => this.openTest(), 'qc-button qc-deeper-test-open');
    this.testButton.title = QC_TEXT.testOpenTooltip; this.testButton.hidden = true;
    this.whereRow = el('div', 'qc-deeper-where-row'); this.whereRow.hidden = true; this.whereRow.append(this.whereButton, this.testButton);
    from.append(this.noneNode, this.foreignHeading, this.chips, this.foreignNone, this.sameLine, this.sourcesBlock, this.whereRow);
    // Where they sit (while the toggle is on): the glow's line (how many likely come from neighbours, and where they
    // sat at the start; the start's hold in its tooltip), a status line, then per source its genes.
    const where = this.whereSection = el('section', 'qc-deeper-where'); where.hidden = true; where.dataset.state = 'off';
    this.glowLegend = el('p', 'qc-deeper-glow-legend'); this.glowLegend.title = glowTip(); this.glowLegend.hidden = true;
    this.glowText = el('span'); this.glowLegend.append(el('i', 'qc-deeper-glow-dot'), this.glowText);
    this.glowUnlit = el('p', 'qc-deeper-muted qc-deeper-glow-unlit'); this.glowUnlit.hidden = true;
    this.whereStatus = el('div', 'qc-deeper-where-status'); this.whereStatus.hidden = true; this.whereList = el('div', 'qc-deeper-where-list');
    where.append(this.heading(T.deepWhereHeading, T.deepWhereIntro), this.glowLegend, this.glowUnlit, this.whereStatus, this.whereList);
    // Its own: the line that names its kind, its top genes and its group in that line's tooltip; the
    // top genes' own line only when no kind is named.
    const own = this.ownSection = el('section', 'qc-deeper-own');
    this.ownKind = el('p'); this.ownGenes = el('p', 'qc-deeper-muted');
    own.append(this.ownKind, this.ownGenes);
    // Loose RNA in the tissue: one line, only from 1 % on.
    const loose = this.looseSection = el('section', 'qc-deeper-loose'); loose.hidden = true;
    this.looseLine = el('p', 'qc-deeper-muted'); loose.append(this.looseLine);
    // All genes (collapsed; virtualised rows of a fixed height).
    const table = this.tableBox = el('details', 'qc-deeper-table');
    this.tableSummary = el('summary'); this.tableScroll = el('div', 'qc-deeper-table-scroll');
    this.table = el('table'); const thead = el('thead'), hr = el('tr');
    for (const [text, cls] of [[T.deepColGene, ''], [T.deepColTranscripts, 'qc-deeper-num'], [T.deepColSplit, ''], [T.deepColNeighbours, 'qc-deeper-num']]) hr.append(el('th', cls, text));
    thead.append(hr); this.tbody = el('tbody'); this.table.append(thead, this.tbody); this.tableScroll.append(this.table);
    this.tableRest = el('p', 'qc-deeper-muted qc-deeper-table-rest'); this.tableRest.hidden = true;
    this.tableNote = el('p', 'qc-deeper-muted', T.deepTableNote);
    table.append(this.tableSummary, this.tableScroll, this.tableRest, this.tableNote);
    // Footer: what it is (its blind spot in the tooltip), How.
    const foot = this.foot = el('footer', 'qc-deeper-foot');
    const estimate = el('p', 'qc-deeper-estimate'); estimate.title = T.deepBlindSpot; estimate.append(el('span', '', T.deepEstimate), this.icon('info'));
    this.how = el('details', 'qc-deeper-how'); this.how.append(el('summary', '', T.deepHowTitle)); this.howList = el('ol'); this.how.append(this.howList);
    foot.append(estimate, this.how);
    scroll.append(from, where, own, loose, table, foot);
    body.append(headline, this.bar, this.legend, scroll);
    root.append(head, this.pinned, this.backButton, this.stateBox, body);
    this.host?.append?.(root);
    // ---- behaviour
    this.findButton.onclick = () => { if (this.cell) this.call('onFind', this.cell); };
    this.smallButton.onclick = () => this.setSmall(!this.small);
    this.closeButton.onclick = () => this.close();
    // Esc closes only while focus is inside; Up/Down move only inside the table.
    this.listen(root, 'keydown', e => {
      if (e.key === 'Escape') { e.preventDefault?.(); e.stopPropagation?.(); this.close(); return; }
      if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && this.tableBox.contains?.(e.target)) { e.preventDefault?.(); e.stopPropagation?.(); this.tableKey(e.key === 'ArrowUp' ? -1 : 1); }
    });
    // Drag the header to move; double-click puts the window back.
    this.listen(head, 'pointerdown', e => {
      if (e.button !== 0 || e.target?.closest?.('button')) return;
      e.preventDefault?.(); try { head.setPointerCapture?.(e.pointerId); } catch { /* not captured */ }
      this.drag = { id: e.pointerId, x: e.clientX, y: e.clientY, left: this.left ?? 0, top: this.top ?? 0, moved: false };
    });
    this.listen(head, 'pointermove', e => {
      const d = this.drag; if (!d || d.id !== e.pointerId) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) < 3) return;
      d.moved = true; this.moveTo(d.left + dx, d.top + dy);
    });
    const release = e => { const d = this.drag; if (!d || (e && d.id !== e.pointerId)) return; this.drag = null; if (d.moved) this.savePlace(); };
    this.listen(head, 'pointerup', release); this.listen(head, 'pointercancel', release); this.listen(head, 'lostpointercapture', () => release(null));
    this.listen(head, 'dblclick', e => { if (e.target?.closest?.('button')) return; this.forgetPlace(); this.place(true); });
    // A pressed control or keyboard focus holds back rebuilding the section it is in.
    for (const node of [this.sources, this.chips, this.tableBox]) {
      this.listen(node, 'pointerdown', () => { this.pressed = true; });
      for (const type of ['pointerup', 'pointercancel', 'pointerleave']) this.listen(node, type, () => { if (this.pressed) { this.pressed = false; if (this.stale) this.later('stale', 0, () => this.rerender()); } });
      this.listen(node, 'focusin', () => { this.focused = node; });
      this.listen(node, 'focusout', () => { if (this.focused === node) this.focused = null; if (this.stale) this.later('stale', 0, () => this.rerender()); });
    }
    this.listen(this.tableBox, 'toggle', () => { if (this.tableBox.open) this.renderTable(true); });
    this.listen(this.tableScroll, 'scroll', () => { if (!this.tableFrame) this.tableFrame = this.clock.requestAnimationFrame(() => { this.tableFrame = null; this.renderTable(false); }); }, { passive: true });
    this.listen(this.sparkCanvas, 'pointermove', e => this.sparkHover(e));
    // The window's size, as the browser reports it (the Mixing legend keeps clear of it; it stays inside the stage).
    if (typeof globalThis.ResizeObserver === 'function') {
      this.observer = new globalThis.ResizeObserver(entries => {
        const box = entries.at(-1)?.borderBoxSize?.[0], rect = entries.at(-1)?.contentRect;
        const w = box?.inlineSize ?? rect?.width ?? 0, h = box?.blockSize ?? rect?.height ?? 0;
        if (w > 0 && h > 0 && (Math.abs(w - this.size.w) > .5 || Math.abs(h - this.size.h) > .5)) { this.size = { w, h }; this.box(); }
      });
      this.observer.observe(root);
    }
  }

  // ---- lifecycle -------------------------------------------------------------------------------------------------------
  /** Another run (or none) is open: the window closes and forgets the old run's answers. */
  reset(run = null) {
    if (this.isOpen) this.close({ focus: false, stop: true });
    for (const key of [...this.timers.keys()]) this.cancel(key);
    this.stopMotion(true);
    this.runId = run?.id ?? null; this.sampleId = run?.sample ?? null; this.posting = null;
    this.mayFollow = false; this.followedAt = -Infinity;
    this.results = new Map(); this.histories = new Map(); this.model = null; this.followed = null; this.asked = null; this.askKey = null;
    this.snapped = new Map();   // cell:frame -> {seq: the counted frame the engine answered it as, counted: that frame's own then}
    this.recounted = new Set();   // cell:frame:count answered again for an answer held on the frame's other count (ensure)
    this.recounting = null;       // ... the one being asked (until its result comes, its ask fails or its promise goes stale)
    this.followFail = null;       // {key: run:cell, tries}: a follow asked for an answer held here failed (followFailed)
    this.modelMemo = null;        // explain's model of the result shown, kept while its inputs are the same (modelOf)
    this.answer = null; this.detail = null; this.error = null; this.shown = null; this.shownKey = null; this.appliedAt = -Infinity;
    this.renderedAt = -Infinity; this.lastSeq = null; this.movedAt = -Infinity; this.workingFor = null; this.unavailable = null;
    this.startRefused = null;   // the engine refused the run's start line: the start kind its refusal named ('none': none)
    this.order = { rows: [], chips: [] }; this.chipsAll = false; this.pin = 0; this.hot = 0; this.focusKey = null; this.back = 0;
    this.rows = []; this.rowNoteFor = 0;
    this.where = null; this.whereWanted = null;   // "Show on the image": off (a closed window turned it off)
    this.glowKey = null; this.kept = null;         // the glow handed to the stage; the suspects last placed (suspects())
    this.farShown = 0;                             // the row the glow lay near none of, as the rows and focus last said
  }
  /** Open on cell `label` (from: the control that opened it; focus returns there on close). */
  open(label, { from = null, focus = true } = {}) {
    const ctx = this.ctx();
    if (!(label > 0) || !ctx.run?.sample) return false;
    if (ctx.run.id !== this.runId) this.reset(ctx.run);
    const was = this.isOpen;
    if (ctx.selected > 0) this.selected = ctx.selected;   // the page's selection as it is now
    this.isOpen = true; this.returnTo = from; this.root.hidden = false;
    if (!was) {
      this.harness?.count('qcDeeperOpened');
      this.log(`opened for ${this.nameOf(label)}`);
      this.small = (ctx.stage?.height ?? 1000) < SHORT_STAGE; this.root.classList?.toggle('qc-deeper-small', this.small);
      this.place(false, label);
    }
    this.target(label, was ? 'select' : 'open');
    if (focus) try { this.root.focus?.({ preventScroll: true }); } catch { /* focus is a convenience */ }
    this.call('onChange');
    return true;
  }
  /** Close: the engine stops following, the stage's outlines, greying out and arrows go, focus returns to the button. */
  close({ focus = true, stop = true } = {}) {
    if (!this.isOpen) return;
    this.isOpen = false; this.root.hidden = true; this.root.dataset.state = 'closed'; this.root.dataset.cell = '';
    for (const key of ['ask', 'apply', 'soon', 'working', 'retry', 'stale-ask', 'recheck', 'stale', 'follow']) this.cancel(key);
    this.recounting = null; this.followFail = null;   // an ask still on its way is asked again when it is wanted again
    this.stopMotion(true);
    // The engine may follow the cell whenever a follow was asked for: answered or not yet, or kept while the sample or its
    // details were learned: it stops following whatever the answer was.
    if (stop && this.runId && (this.followed || this.posting || this.mayFollow)) this.api.qcDeeperStop?.(this.runId)?.catch?.(() => {});
    this.mayFollow = false; this.followedAt = -Infinity;
    this.followed = null; this.posting = null; this.asked = null; this.askKey = null; this.pin = 0; this.hot = 0; this.back = 0;
    this.shown = null; this.shownModel = null; this.shownKey = null; this.boxKey = null; this.cell = 0;
    this.rows = []; this.rowNoteFor = 0; setHidden(this.rowNote, true);
    this.body.hidden = true; this.stateBox.hidden = true; this.chipNodes?.clear(); this.rowNodes?.clear(); this.order = { rows: [], chips: [] };
    this.setFocus(null); this.call('onBox', null);
    this.whereOff('closed'); this.whereWanted = null; this.paintWhereRow();   // after the focus: no arrows come back
    this.log('closed');
    const to = this.returnTo; this.returnTo = null;
    if (focus && to?.isConnected !== false) try { to?.focus?.(); } catch { /* the button may be gone */ }
    this.call('onChange');
  }
  toggle(label, options = {}) {
    if (this.isOpen && (!label || label === this.cell)) { this.close(); return false; }
    if (this.isOpen && label) { this.target(label, 'select'); return true; }
    return this.open(label, options);
  }
  /** The page's selection changed (stage, UMAP, most active cells): the window follows it; a cleared selection keeps the
   *  window on its last cell (still outlined), and says so. */
  select(label) {
    label = label > 0 ? label : 0;
    if (label === this.selected) return;
    this.selected = label;
    if (!this.isOpen) return;
    if (label && label !== this.cell) this.target(label, 'select');
    else { this.paintPinned(); this.paintWhereRow(); }   // whether the page can test this cell follows the selection
  }
  /** Look at another cell: the old content dims until the new answer lands (no skeleton, no jump). */
  target(label, via = 'select') {
    if (!(label > 0)) return;
    const same = label === this.cell;
    if (same && via !== 'open') return;
    this.back = via === 'neighbour' ? this.cell : 0;   // one step back only
    if (!same && this.cell && via !== 'open') this.harness?.count('qcDeeperRetargets');
    this.cell = label; this.chipsAll = false; this.pin = 0; this.hot = 0; this.order = { rows: [], chips: [] };
    this.rowNoteFor = 0; setHidden(this.rowNote, true);
    this.error = null; this.cancel('retry'); this.workingFor = null; this.cancel('follow'); this.followFail = null;
    this.root.dataset.cell = this.nameOf(label);
    this.retargetAt = this.now();
    if (this.shown) this.body.classList?.add('qc-deeper-dim');
    this.later('working', WORKING_MS, () => { this.workingFor = this.cell; this.update(true); });
    this.loadHistory(label);
    this.followed = null; this.posting = null;   // the engine follows the new cell from the next request
    this.setFocus(null);   // the old cell's arrows retract; the new ones draw on when its answer lands
    this.whereOff('another cell'); this.whereWanted = null; this.paintWhereRow();   // the dots go with the old cell
    this.paintHeader(''); this.paintPinned(); this.paintBack();
    this.ensure(true);
    this.update(true);
    this.call('onChange');
  }
  goBack() { const to = this.back; if (!to) return; this.target(to, 'back'); this.call('onSelect', to); }
  /** The socket reconnected (the engine may have restarted): ask again. */
  refresh() {
    this.followed = null; this.posting = null; this.asked = null; this.askKey = null; this.cancel('follow'); this.followFail = null;
    if (this.isOpen) { this.ensure(true); this.update(true); }
  }
  /** A full-screen view covers the page (main.js: Compare's dialog), or no longer does. Covered, nothing in the window
   *  moves or changes: a motion on its way lands at once (as when it closes), and what would be shown meanwhile (a pushed
   *  result, a timer's "Working out…") waits; uncovered (`catchUp`: unless another run or the picker opens instead), the
   *  window shows what is wanted then, once. */
  setCovered(on, catchUp = true) {
    on = !!on;
    if (on === this.covered) return;
    this.covered = on;
    if (on) { this.stopMotion(true); return; }
    const missed = this.coveredMissed; this.coveredMissed = false;
    if (missed && catchUp) this.update(true);
  }
  /** The check's view or the run changed, or a snapshot was painted (main.js renderQc, in the paint batch). */
  changed() {
    if (!this.isOpen) return;
    const ctx = this.ctx();
    if (ctx.run?.id !== this.runId) { this.reset(ctx.run ?? null); return; }
    if (ctx.seq !== this.lastSeq) { if (this.lastSeq != null && !ctx.live) this.movedAt = this.now(); this.lastSeq = ctx.seq; }
    this.ensure(false);
    this.update(false);
    this.paintBasis();   // the frame's own statistics (its background), whether or not the split changed
    this.paintWhereRow();   // and whether the page can test this cell now (context testable)
    // The run came to rest (or stopped playing) with nothing else changed: the dots' frame asked for itself (whereRest).
    const wanted = this.where?.wanted;
    if (wanted && this.whereRest(wanted, ctx)) this.whereAskSoon(wanted, ctx);
  }
  frame() { this.changed(); }
  /** A frame's cell statistics arrived (main.js CellStats): the basis's tooltip (what proseg reads as background in the
   *  result's frame) follows them. */
  statsChanged() { if (this.isOpen) this.paintBasis(); }

  // ---- requests ---------------------------------------------------------------------------------------------------------
  /** Whether the run's check can answer now (on and ready: anything else is the check's own state). */
  requestable(ctx = this.ctx()) {
    const view = ctx.qc;
    return !!ctx.run?.sample && view?.enabled !== false && READY_VIEWS.has(view?.status) && !this.unavailable;
  }
  wantedSeq(ctx = this.ctx()) { return ctx.seq === 0 ? 'start' : Number.isInteger(ctx.seq) && ctx.seq > 0 ? ctx.seq : null; }
  /** Whether the run has no start line (noStartLine: the context's start kind, else the one the engine's refusal of
   *  the start line named), and why; null while it has one. */
  noStart(ctx = this.ctx()) {
    return noStartLine(ctx.startSource, typeof ctx.startKind === 'string' ? ctx.startKind : this.startRefused);
  }
  /** Ask the engine for what the frame on screen needs: once per cell while live (it follows the cell and pushes results
   *  with its passes), or that frame (debounced) while paused, stepped or scrubbed; nothing for a frame held here, nor
   *  for the start line of a run without one. */
  ensure(now = false) {
    const ctx = this.ctx();
    if (!this.isOpen || !this.cell || !this.runId || !this.requestable(ctx)) return;
    const seq = this.wantedSeq(ctx);
    if (seq == null || (seq === 'start' && this.noStart(ctx) != null)) return;
    if (this.detail && !now) return;   // the details are missing or being learned: asked again when they are there
    if (this.timers.has('recheck') && !now) return;   // the engine said its check is not ready: asked again by the timer
    const cell = this.cell - 1, active = ['running', 'paused'].includes(ctx.run?.state), key = `${cell}:${seq}`;
    const follow = `${this.runId}:${cell}`;
    // A failure is asked again after its wait (2, 5, 10, 30 s), not with every snapshot.
    if (this.error && this.timers.has('retry') && !now && (ctx.live || this.error.key === key)) return;
    if (ctx.live) {
      // Followed: the engine pushes this cell's results with its passes (a promise that goes unanswered is asked again).
      // The follow is a lease the engine ends when nobody renews it: asked again every FOLLOW_RENEW_MS.
      if (this.posting === follow || (this.followed === follow && this.now() - this.followedAt < FOLLOW_RENEW_MS)) return;
      this.post({ seq, cell, follow: true }); return;
    }
    const at = this.answeredAt(cell, seq, ctx), held = this.results.get(cell)?.get(at);
    // Held already (pushed for another ask, or kept from before the window opened) on the count the frame reads now: a
    // frame counted again for a pause, the hold, a finish or the end (context sampled) reads the count proseg saves
    // there, every other one the averaged count.
    const want = seq === 'start' ? 'averaged' : ctx.sampled?.(at) === true ? 'sampled' : 'averaged';
    if (held && (held.count ?? 'averaged') === want) {
      // On a run that samples or rests the engine is still asked once to follow the window's cell, or the box its dots
      // arm while paused would wait for no one and the first reply after Resume come out approximate (SPEC D.1). A
      // follow only: its failure is not the frame's (the answer held stays as it is, no error said), asked again after
      // its wait (followFailed).
      if (active && this.followed !== follow && this.posting !== follow && !this.timers.has('follow')) this.post({ seq, cell, follow: true }, null, true);
      return;
    }
    if (held) {
      // Held on the other count (the frame was counted again after its answer came, while nothing was followed): asked
      // for once per cell, frame and count (one at a time, a promise of it kept until its result comes; asked again only
      // when no answer came: a failure, a promise gone stale), which the engine answers on the frame's count now; an
      // engine that still answers the other one (its file gone) is not asked again, and the answer held stays.
      const again = `${cell}:${at}:${want}`;
      if (!this.recounted.has(again) && this.recounting !== again) this.post({ seq, cell, follow: active }, again);
      return;
    }
    if (this.asked === key || (this.askKey === key && this.timers.has('ask'))) return;
    this.askKey = key;
    const go = () => { if (this.askKey === key && this.isOpen) this.post({ seq, cell, follow: active }); };
    if (now) { this.cancel('ask'); go(); } else this.later('ask', ASK_MS, go);
  }
  /** Ask the engine (POST /runs/{id}/qc/deeper). `again`: the ask is ensure's for an answer held on the frame's other
   *  count (cell:frame:count), which a result answers for good (recounted; take and event). `only`: a follow asked for
   *  an answer held on its own count, whose failure leaves the window as it is (followFailed). */
  async post(body, again = null, only = false) {
    const run = this.runId, key = `${body.cell}:${body.seq}`;
    if (!run || typeof this.api.qcDeeper !== 'function') return;
    this.asked = key; this.askKey = null;
    const follow = body.follow ? `${run}:${body.cell}` : null;
    if (follow) { this.posting = follow; this.mayFollow = true; }
    if (again) this.recounting = again;
    let answer;
    try { answer = await this.api.qcDeeper(run, body); }
    catch (error) {
      if (follow && this.posting === follow) this.posting = null;
      if (again && this.recounting === again) this.recounting = null;
      if (run !== this.runId || !this.isOpen) return;
      this.asked = null;
      if (only) { this.followFailed(follow, error); return; }
      // Where Clusters and the UMAP cannot run the engine refuses with 503 and reason "analysis" (engine/deeper.py):
      // the window says that. Any other 503 (a storage error, the engine shutting down) is a failure, asked again.
      if (error?.status === 503 && error?.data?.reason === 'analysis') { this.unavailable = 'analysis'; this.update(true); return; }
      // A run without a start line (its cells started with nuclei: engine/deeper.py, reason "no_start"): its first
      // frame says so, as for a run from nuclei only, and the start line is not asked again.
      if (body.seq === 'start' && error?.status === 404 && error?.data?.reason === 'no_start') {
        const kind = error.data.start_kind;
        this.startRefused = typeof kind === 'string' && kind !== 'cells' ? kind : 'none'; this.update(true); return;
      }
      this.fail(key, !error?.status, error?.status ?? null); return;
    }
    if (follow && this.posting === follow) this.posting = null;
    const open = run === this.runId && this.isOpen;
    // A promise of the answer asked again (working) keeps the ask on until its result comes (event) or the promise goes
    // stale (take's stale-ask): each repaint meanwhile asks nothing more.
    if (again && this.recounting === again && !(open && answer?.state === 'working')) this.recounting = null;
    if (!open) return;
    if (again && answer?.state === 'ready' && answer.result) this.recounted.add(again);
    this.take(answer, body);
  }
  /** A follow asked for an answer held here (ensure's `only`) failed: not the frame's failure, so the answer stays as it
   *  is and no error is said; the follow is asked again after the waits a failure gets (2, 5, 10, then every 30 s). */
  followFailed(follow, error) {
    const tries = (this.followFail?.key === follow ? this.followFail.tries : 0) + 1;
    this.followFail = { key: follow, tries };
    this.log(`could not ask the engine to follow ${follow}: ${error?.status ? `HTTP ${error.status}` : 'no connection'} (try ${tries})`);
    this.later('follow', RETRY_MS[Math.min(tries, RETRY_MS.length) - 1], () => this.ensure(true));
  }
  /** The engine's answer to a request (6.4): a result, a promise of one, or the state the check or the details are in.
   *  A frame without proseg's count of its own (between counted frames, or its file gone) is answered as the counted
   *  frame the engine snaps it to, whose seq the answer names: its result is kept as that frame's, and that frame's
   *  result is the one for the frame asked (answeredAt), so it is not asked for again. */
  take(answer, body) {
    if (!answer || typeof answer !== 'object') { this.fail(`${body.cell}:${body.seq}`, false, null); return; }
    const key = `${body.cell}:${body.seq}`;
    const at = Number.isInteger(answer.seq) && Number.isInteger(body.seq) && answer.seq !== body.seq ? answer.seq : null;
    if (at != null) {
      this.snapped.delete(key); this.snapped.set(key, { seq: at, counted: this.ctx().counted?.(body.seq) === true });
      if (this.snapped.size > SNAPPED_KEPT) this.snapped.delete(this.snapped.keys().next().value);
    }
    if (answer.model) this.model = answer.model;
    if (body.follow && ['ready', 'working'].includes(answer.state)) { this.followed = `${this.runId}:${body.cell}`; this.followedAt = this.now(); this.followFail = null; }
    if (this.error?.key === key) { this.error = null; this.cancel('retry'); }
    this.answer = { state: answer.state, reason: answer.reason ?? null, progress: answer.progress ?? null, eta_s: answer.eta_s ?? null, key, at: this.now() };
    if (['needs_detail', 'detail', 'detail_failed'].includes(answer.state)) {
      this.detail = { state: answer.state, reason: answer.reason ?? null, progress: answer.progress ?? null, eta_s: answer.eta_s ?? null };
      this.asked = null;
    } else if (this.detail && !CHECK_STATES.has(answer.state)) this.detail = null;
    if (answer.state === 'ready' && answer.result) { if (!this.store(answer.result, { seq: at ?? body.seq, cell: body.cell, model: answer.model }) || at != null) this.asked = null; }
    else if (answer.state === 'working') this.later('stale-ask', STALE_MS, () => { this.recounting = null; if (this.asked === key) { this.asked = null; this.followed = null; this.ensure(true); } });
    else {
      this.asked = null;
      // The engine's check is not ready for this yet (its own events say when the sample is): asked again in a while.
      if (CHECK_STATES.has(answer.state)) { this.followed = null; this.later('recheck', RETRY_MS[0], () => { if (this.isOpen && this.requestable()) this.ensure(true); }); }
    }
    this.update(true);
  }
  /** A request or a computation failed: said in plain words, asked again after 2, 5, 10 and 30 s (a lost connection
   *  says that it tries again by itself). */
  fail(key, connection = false, status = null) {
    const tries = (this.error?.key === key ? this.error.tries : 0) + 1;
    this.error = { key, tries, connection, status };
    this.log(`could not work out ${key}: ${connection ? 'no connection' : status ? `HTTP ${status}` : 'error'} (try ${tries})`);
    if (status !== 422) this.later('retry', RETRY_MS[Math.min(tries, RETRY_MS.length) - 1], () => { this.asked = null; this.askKey = null; this.followed = null; this.ensure(true); });
    this.update(true);
  }
  retry() { this.cancel('retry'); this.error = null; this.asked = null; this.askKey = null; this.followed = null; this.ensure(true); this.update(true); }
  /** A pushed {ev: "deeper"} event: a result, an error, or how the sample's details are being learned. */
  event(m) {
    if (!m || typeof m !== 'object') return;
    if (m.run === undefined && m.sample !== undefined) {
      if (m.sample !== this.sampleId) return;
      if (m.state === 'ready') { this.detail = null; this.answer = null; this.asked = null; this.followed = null; this.ensure(true); }
      else if (m.state === 'detail' || m.state === 'detail_failed') this.detail = { state: m.state, reason: m.reason ?? null, progress: m.progress ?? null, eta_s: m.eta_s ?? null };
      this.update(true);
      return;
    }
    if (!this.runId || m.run !== this.runId) return;
    const cell = Number.isInteger(m.cell) ? m.cell : m.result?.cell;
    if (m.state === 'error') {
      if (this.recounting?.startsWith(`${cell}:${m.seq}:`)) this.recounting = null;   // asked again after the failure's wait
      if (!this.isOpen || cell !== this.cell - 1) return;
      const seq = m.seq === 'start' ? 'start' : m.seq, wanted = this.wantedSeq();
      this.log(`the engine could not work out ${cell}:${seq}: ${typeof m.error === 'string' ? m.error : 'error'}`);
      const mine = seq === wanted || (wanted != null && seq === this.answeredAt(cell, wanted));   // the frame on screen's
      if (mine || this.asked === `${cell}:${seq}`) { this.asked = null; this.fail(`${cell}:${mine ? wanted : seq}`); }
      return;
    }
    if (!m.result) return;
    const stored = this.store(m.result, { seq: m.seq, iter: m.iter, cell, model: m.model });
    // The result of a frame asked again for an answer held on its other count (ensure): answered for good.
    if (stored && this.recounting?.startsWith(`${stored.cellIndex}:${stored.seq}:`)) { this.recounted.add(this.recounting); this.recounting = null; }
    if (!stored || !this.isOpen || stored.cellIndex !== this.cell - 1) return;
    // The answer to the frame asked, or to the counted frame the engine answers that one as: nothing is asked again.
    if (this.asked === `${stored.cellIndex}:${stored.seq}` || (this.asked?.startsWith(`${stored.cellIndex}:`)
      && this.snapped.get(this.asked)?.seq === stored.seq)) this.asked = null;
    this.soon();
  }
  /** A result to keep (per cell, the newest KEEP_RESULTS; KEEP_CELLS cells): returns {cellIndex, seq} or null. */
  store(raw, { seq, iter, cell, model } = {}) {
    const r = normalise({ ...raw, seq: raw?.seq ?? seq, iter: raw?.iter ?? iter, cell: raw?.cell ?? cell, model: raw?.model ?? model });
    if (!r) return null;
    const cellIndex = Number.isInteger(r.cell) ? r.cell : Number.isInteger(cell) ? cell : null;
    if (cellIndex == null) return null;
    if (r.model && this.model && r.model !== this.model) { this.results.clear(); this.histories.clear(); }
    if (r.model) this.model = r.model;
    let held = this.results.get(cellIndex);
    if (!held) { held = new Map(); this.results.set(cellIndex, held); }
    else { this.results.delete(cellIndex); this.results.set(cellIndex, held); }
    // The same answer twice (a request's answer and its push): the one held stays, so nothing counts as a new update.
    const had = held.get(r.seq);
    if (had && signature(had) === signature(r)) return { cellIndex, seq: r.seq };
    held.set(r.seq, r);
    if (held.size > KEEP_RESULTS) for (const k of held.keys()) { if (held.size <= KEEP_RESULTS) break; if (k !== this.shown?.result?.seq) held.delete(k); }
    for (const k of this.results.keys()) { if (this.results.size <= KEEP_CELLS) break; if (k !== this.cell - 1) this.results.delete(k); }
    // A cell's small line goes with its results (kept for every cell ever visited).
    for (const k of this.histories.keys()) if (!this.results.has(k) && k !== this.cell - 1 && k !== cellIndex) this.histories.delete(k);
    // The small line grows with every result of its cell (at most HISTORY_KEPT_POINTS, the oldest live points going); an
    // answer's byte replaces the one at its iteration (a frame counted again: the count proseg saves there).
    const h = this.histories.get(cellIndex);
    if (h && r.state === 'ready' && finite(r.iter) && r.seq !== 'start') {
      h.points = h.points.filter(([i]) => i !== r.iter);
      h.points.push([r.iter, r.code]); h.points.sort((a, b) => a[0] - b[0]);
      if (h.points.length > HISTORY_KEPT_POINTS) h.points.splice(0, h.points.length - HISTORY_KEPT_POINTS);
    }
    return { cellIndex, seq: r.seq };
  }
  /** A pushed result: applied with the next snapshot's paint while live (at most WITH_SNAPSHOT_MS later), else now. */
  soon() {
    if (!this.batch || !this.ctx().live) { this.update(true); return; }
    this.batch.ride?.('deeper', () => this.update(false));
    if (!this.timers.has('soon')) this.later('soon', WITH_SNAPSHOT_MS, () => this.batch.mark?.('deeper', () => this.update(false)));
  }
  /** The cell's bytes over the run, for the small line (GET /runs/{id}/qc/deeper/history), once per cell. */
  loadHistory(label) {
    const cell = label - 1, run = this.runId;
    if (!run || this.histories.has(cell) || typeof this.api.qcDeeperHistory !== 'function') return;
    this.histories.set(cell, { points: [], start: null, loading: true });
    this.api.qcDeeperHistory(run, cell, HISTORY_POINTS).then(answer => {
      if (run !== this.runId) return;
      const h = this.histories.get(cell); if (!h) return;
      const points = (Array.isArray(answer?.points) ? answer.points : []).filter(p => Array.isArray(p) && finite(p[0]) && Number.isInteger(p[1]));
      const seen = new Set(points.map(p => p[0]));
      for (const p of h.points) if (!seen.has(p[0])) points.push(p);
      points.sort((a, b) => a[0] - b[0]);
      Object.assign(h, { points, start: Number.isInteger(answer?.start) ? answer.start : null, loading: false });
      if (this.isOpen && this.cell - 1 === cell) { this.shownKey = null; this.update(true); }
    }).catch(() => { const h = this.histories.get(cell); if (h) h.loading = false; });
  }
  /** "Learn the details" (POST /samples/{id}/qc {detail: true}). */
  async learnDetail() {
    const sample = this.sampleId;
    if (!sample || typeof this.api.learnSampleDetail !== 'function') return;
    this.log('learn the details asked for');
    this.detail = { state: 'detail', progress: null, eta_s: null, reason: null }; this.update(true);
    try {
      const answer = await this.api.learnSampleDetail(sample);
      if (sample !== this.sampleId) return;
      const d = answer?.detail;
      if (answer?.model?.deep === true || d?.state === 'ready') { this.detail = null; this.answer = null; this.followed = null; this.ensure(true); }
      else if (d && typeof d === 'object') this.detail = { state: d.state === 'failed' ? 'detail_failed' : d.state === 'none' ? 'needs_detail' : 'detail', progress: d.progress ?? null, eta_s: d.eta_s ?? null, reason: d.reason ?? null };
    } catch { if (sample === this.sampleId) this.detail = { state: 'detail_failed', reason: null, progress: null, eta_s: null }; }
    this.update(true);
  }

  // ---- what to show -------------------------------------------------------------------------------------------------------
  /** The newest result at or before the frame on screen for the window's cell (never a later one). */
  pick(ctx = this.ctx()) {
    const held = this.results.get(this.cell - 1);
    if (!held) return null;
    const key = newestAtOrBefore(held.keys(), ctx.seq);
    return key == null ? null : held.get(key);
  }
  /** The frame the engine answers frame `seq` of cell `cell` (label - 1) with: the counted frame its answer named
   *  (take: one without proseg's count of its own is answered as the nearest counted frame), else that frame. One that
   *  was not counted then and is now (context counted: a pause's count can come after an answer) is its own again. */
  answeredAt(cell, seq, ctx = this.ctx()) {
    const key = `${cell}:${seq}`, snap = this.snapped.get(key);
    if (!snap) return seq;
    if (!snap.counted && ctx.counted?.(seq) === true) { this.snapped.delete(key); return seq; }
    return snap.seq;
  }
  /** The state to show, in their order of priority. */
  status(ctx = this.ctx()) {
    const view = ctx.qc, run = ctx.run;
    if (!run?.sample) return { state: 'unavailable', text: cantRunText(QC_TEXT.unavailableTable) };
    if (this.unavailable) return { state: 'unavailable', text: cantRunText(reasonText('unavailable', this.unavailable)) };
    if (view?.enabled === false || view?.status === 'off') return { state: 'off' };
    // A run without proseg's own count, or an old proseg-live: why, or how to update it, in the line's tooltip.
    const tip = reason => unavailableTooltip(reason, { prosegFix: ctx.prosegFix ?? null, notebook: !!ctx.notebook });
    if (view?.status === 'unavailable') return { state: 'unavailable', text: cantRunText(reasonText('unavailable', view.reason)), tip: tip(view.reason) };
    if (view?.status === 'waiting') return { state: 'waiting', view };
    if (view?.status === 'learning') return { state: 'learning', view };
    if (view?.status === 'failed') return { state: 'failed', view };
    if (!view || view.status === 'loading') return view?.failed ? { state: 'retrying' } : { state: 'loading' };
    if (this.detail) return { state: this.detail.state, detail: this.detail, view };
    if (this.answer && CHECK_STATES.has(this.answer.state) && !this.pick(ctx) && this.now() - this.answer.at < 2 * RETRY_MS[0]) {
      const a = this.answer;
      if (a.state === 'off') return { state: 'off' };
      if (a.state === 'unavailable') return { state: 'unavailable', text: cantRunText(reasonText('unavailable', a.reason)), tip: tip(a.reason) };
      return { state: a.state, view: { ...view, progress: a.progress ?? view.progress, eta_s: a.eta_s ?? view.eta_s, reason: a.reason ?? view.reason } };
    }
    const seq = this.wantedSeq(ctx), none = seq === 'start' ? this.noStart(ctx) : null;
    if (none != null) return { state: 'nuclei_only', text: T[(ctx.buttons?.start && ctx.buttons?.step ? NO_START_TEXT : NO_START_ENDED_TEXT)[none]] };
    const result = this.pick(ctx);
    const pending = result && result.seq !== this.answeredAt(this.cell - 1, seq, ctx);
    if (this.error && (!result || this.error.key === `${this.cell - 1}:${seq}`)) return { state: this.error.connection ? 'retrying' : 'error', result };
    if (!result) return { state: 'working', working: this.workingFor === this.cell };
    return { state: result.state, result, pending, seq };
  }
  /** Show what is wanted now: at most one new result a second while live (in a snapshot's paint batch), at once
   *  otherwise. `now`: the user's own action or a paused run (no throttle). */
  update(now = false) {
    if (!this.isOpen) return;
    if (this.covered) { this.coveredMissed = true; return; }   // under Compare: shown once it closes (setCovered)
    const ctx = this.ctx(), st = this.status(ctx);
    const tag = this.frameTagOf(st, ctx);
    const key = [st.state, st.result ? `${st.result.cell}:${st.result.seq}:${serialOf(st.result)}` : '', st.text ?? '', st.tip ?? '', tag, st.working ? 1 : 0, this.chipsAll,
      st.view ? [st.view.progress?.step, st.view.progress?.of, st.view.eta_s, st.view.reason, st.view.shared, st.view.relearn, st.view.sampleModel?.seconds] : '',
      st.detail ? [st.detail.state, st.detail.progress?.step, st.detail.progress?.of, st.detail.eta_s, st.detail.reason] : '',
      ctx.run?.state, this.cell].join('|');
    if (key === this.shownKey) return;
    const newer = st.result && this.shown?.result && st.result !== this.shown.result && st.result.cell === this.shown.result.cell;
    if (newer && ctx.live && !now) {
      const wait = APPLY_MS - (this.now() - this.appliedAt);
      if (wait > 0) { if (!this.timers.has('apply')) this.later('apply', wait, () => this.soon()); return; }
    }
    this.cancel('apply');
    this.render(st, ctx, tag, key);
  }
  labelOf(cell) {
    if (Number.isInteger(cell) && cell >= 0) return cell + 1;
    if (typeof cell === 'string') { const i = (this.ctx().cells || []).indexOf(cell); return i >= 0 ? i + 1 : 0; }
    return 0;
  }
  cellName(cell) { const label = this.labelOf(cell); return label ? this.nameOf(label) : String(cell ?? ''); }
  frameTagOf(st, ctx) {
    if (!st.result) return '';
    const r = st.result, run = ctx.run, frames = ctx.frames || [], newest = frames.length - 1;
    if (r.seq === 'start') return frameTag({ kind: 'start', startSource: ctx.startSource });
    if (st.pending && !ctx.live && Number.isInteger(ctx.seq)) return frameTag({ kind: 'earlier', iter: r.iter, asked: frames[ctx.seq]?.iter ?? ctx.seq });
    // The frame on screen answered as the nearest counted frame (it has no proseg count of its own), said so.
    if (!ctx.live && Number.isInteger(ctx.seq) && r.seq !== ctx.seq) return frameTag({ kind: 'snapped', iter: r.iter });
    if (['done'].includes(run?.state)) return frameTag({ kind: 'finished', iter: r.iter });
    if (['stopped', 'failed'].includes(run?.state)) return frameTag({ kind: 'stopped', iter: r.iter });
    if (ctx.live) return frameTag({ kind: 'live', iter: r.iter });
    if (Number.isInteger(ctx.seq) && ctx.seq < newest) return frameTag({ kind: 'older', iter: r.iter });
    return frameTag({ kind: 'paused', iter: r.iter });
  }

  // ---- rendering ----------------------------------------------------------------------------------------------------------
  render(st, ctx, tag, key) {
    const t0 = this.now(), r = st.result ?? null;
    const isNew = r && r !== this.shown?.result;
    const startKind = typeof ctx.startKind === 'string' ? ctx.startKind : this.startRefused;
    const model = r ? this.modelOf(r, ctx, startKind) : null;
    // Motion only for a change at least MOTION_GAP_MS after the previous one, while live or in a scrub's transition.
    const moving = (ctx.live || this.now() - this.movedAt < MOVE_WINDOW_MS) && !this.reducedMotion && !this.doc?.hidden && !this.covered;
    const motion = !!isNew && !!this.shownModel && this.shown?.result?.cell === r?.cell && this.shownModel.state === 'ready'
      && model?.state === 'ready' && moving && this.now() - this.appliedAt >= MOTION_GAP_MS;
    this.shownKey = key;
    this.root.dataset.state = st.state;
    this.paintHeader(tag, st);
    const result = model?.state === 'ready';
    if (st.state === 'ready' || (result && ['error', 'retrying'].includes(st.state))) {
      this.stateBox.hidden = st.state === 'ready'; if (!this.stateBox.hidden) this.paintStatus(st, ctx);
      this.body.hidden = false; this.body.classList?.remove('qc-deeper-dim');
      this.paintResult(model, r, ctx, motion);
    } else if (st.state === 'working' && this.shownModel?.state === 'ready') {
      // The old content stays, dimmed, until the new answer lands.
      this.body.classList?.add('qc-deeper-dim'); this.stateBox.hidden = !st.working; if (st.working) this.paintStatus(st, ctx);
    } else {
      this.body.hidden = true; this.body.classList?.remove('qc-deeper-dim');
      this.stateBox.hidden = false; this.paintStatus({ ...st, model }, ctx);
      if (st.state !== 'working') {
        this.shownModel = model ?? null; this.setFocus(null);
        this.whereOff('no split to show'); this.whereWanted = null;
      }
    }
    // Every state: Show on the image and "Test a setting here…" in their row, or the entry beside the check's switch
    // while it is off (a state box painted again leaves it out until then).
    this.paintWhereRow();
    this.shown = { state: st.state, result: r, tag };
    if (isNew) { this.appliedAt = this.now(); this.harness?.count('qcDeeperApplied'); }
    this.renderedAt = this.now();
    this.box();
    try { globalThis.performance?.measure?.('plv:qc-deeper-apply', { start: t0, detail: { state: st.state, seq: r?.seq ?? null } }); globalThis.performance?.clearMeasures?.('plv:qc-deeper-apply'); }
    catch { /* a profiling aid only */ }
  }
  /** explain's model of result `r` as the window shows it, kept while nothing it reads changes (the result itself, the
   *  cell's start byte, how the run started and its kept constants, the neighbours' labels and names, which come with
   *  the run's cells after it opens, and their cell groups, which Clusters & UMAP move): a frame tag or the run's state
   *  changing paints the same result again without working it out again (P5). */
  modelOf(r, ctx, startKind) {
    const ready = r.state === 'ready', startCode = ready ? this.startCode() : null, kept = ready ? ctx.run?.constants?.kept ?? [] : [];
    const near = ready ? r.neighbours.map(n => {
      const l = this.labelOf(n.cell), g = l ? this.o.group?.(l) ?? null : null;
      return [l, this.cellName(n.cell), g ? [g.label, g.color, g.iter] : null];
    }) : null;
    const key = JSON.stringify([serialOf(r), startCode, ctx.startSource ?? null, startKind ?? null, kept, near]);
    if (this.modelMemo?.key === key) return this.modelMemo.model;
    const model = ready
      ? explain(r, { name: c => this.cellName(c), label: c => this.labelOf(c), group: l => this.o.group?.(l) ?? null,
        startCode, start: r.seq === 'start', startSource: ctx.startSource, startKind, kept })
      : explain(r, { startSource: ctx.startSource, startKind });
    this.modelMemo = { key, model };
    return model;
  }
  startCode() {
    const h = this.histories.get(this.cell - 1);
    if (Number.isInteger(h?.start)) return h.start;
    const v = this.o.startCode?.(this.cell);
    return Number.isInteger(v) ? v : null;
  }
  paintHeader(tag = this.shown?.tag ?? '', st = null) {
    const name = this.nameOf(this.cell);
    setText(this.title, name);
    const color = this.o.swatch?.(this.cell) ?? COLORS.own;
    if (this.swatch.dataset.color !== color) { this.swatch.dataset.color = color; this.swatch.style.background = color; }
    this.root.setAttribute('aria-label', deep('deepWindowLabel', { cell: name || '…' }));
    setText(this.frameNode, st?.state === 'working' ? '' : tag);
    this.setIcon(this.smallButton._icon, this.small ? 'expand' : 'minimise');
    const label = this.small ? T.deepBigger : T.deepSmaller;
    if (this.smallButton.title !== label) { this.smallButton.title = label; this.smallButton.setAttribute('aria-label', label); }
  }
  paintPinned() { setHidden(this.pinned, !(this.isOpen && this.cell && !this.selected)); }
  paintBack() {
    setHidden(this.backButton, !this.back);
    if (this.back) setText(this.backButton, deep('deepBack', { cell: this.nameOf(this.back) }));
  }
  /** The non-result states, rebuilt only when their words change. */
  paintStatus(st, ctx) {
    const view = st.view ?? ctx.qc ?? {};
    const k = JSON.stringify([st.state, st.text, st.tip, st.working, view.progress, view.eta_s, view.reason, view.shared, view.relearn, st.detail,
      st.model?.lines, this.cell, ctx.startSource, view.room ?? null]);
    if (this.stateBox._key === k) return;
    this.stateBox._key = k; this.stateBox.replaceChildren();
    const p = (text, cls = '') => { if (!text) return null; const node = this.el('p', cls, text); this.stateBox.append(node); return node; };
    const bar = (step, of, label) => {
      const prog = this.el('progress', 'qc-progress'); prog.max = of; prog.value = step; prog.setAttribute('aria-label', label); this.stateBox.append(prog);
    };
    switch (st.state) {
      case 'unavailable': { const line = p(st.text); if (line && st.tip) line.title = st.tip; break; }
      case 'off': p(T.deepOff); this.stateBox.append(this.offRow); break;   // its switch (and the test's entry: paintWhereRow)
      case 'waiting': p(reasonText('waiting', view.reason, view.room)); if (view.shared) p(QC_TEXT.cardShared); p(T.deepLearningNote, 'qc-deeper-muted'); break;
      case 'learning': {
        const of = Math.max(1, view.progress?.of ?? 50), step = clamp(view.progress?.step ?? 0, 0, of);
        const line = finite(view.eta_s) ? qcCopy(of === 50 ? 'learning' : 'learningProgress', { k: number(step), of: number(of), t: duration(view.eta_s) })
          : qcCopy('learningNoTime', { k: number(step), of: number(of) });
        p(line); bar(step, of, line);
        p(view.relearn ? qcCopy('softwareChanged', { t: duration(view.eta_s) }) : T.deepLearningNote, 'qc-deeper-muted');
        if (view.shared) p(QC_TEXT.cardShared, 'qc-deeper-muted');
        break;
      }
      case 'failed':
        p(qcCopy('failed', { reason: reasonText('failed', view.reason, view.room) }));
        this.stateBox.append(this.button(QC_TEXT.tryAgain, 'retry-learning', () => this.call('onRetryLearning')));
        break;
      case 'needs_detail': {
        const seconds = view.sampleModel?.seconds;
        p(finite(seconds) && seconds > 0 ? deep('deepNeedsDetail', { t: duration(seconds) }) : T.deepNeedsDetailNoTime);
        this.stateBox.append(this.button(T.deepLearnDetail, 'learn-detail', () => this.learnDetail()));
        break;
      }
      case 'detail': {
        const d = st.detail ?? {}, of = Math.max(1, d.progress?.of ?? 50), step = clamp(d.progress?.step ?? 0, 0, of);
        const line = finite(d.eta_s) ? deep('deepDetailLearning', { k: number(step), of: number(of), t: duration(d.eta_s) })
          : deep('deepDetailLearningNoTime', { k: number(step), of: number(of) });
        p(line); bar(step, of, line); p(T.deepLearningNote, 'qc-deeper-muted');
        break;
      }
      case 'detail_failed': {
        const d = st.detail ?? {};
        p(detailFailedText(d.reason, d.reason ? reasonText('failed', d.reason) : null));
        this.stateBox.append(this.button(QC_TEXT.tryAgain, 'learn-detail', () => this.learnDetail()));
        break;
      }
      case 'nuclei_only': p(st.text ?? T.deepNucleiOnlyRun); break;   // no start line: nuclei only, or with cells
      case 'working': if (st.working) p(deep('deepWorking', { cell: this.nameOf(this.cell) }), 'qc-deeper-muted'); break;
      case 'error': p(T.deepError); this.stateBox.append(this.button(QC_TEXT.tryAgain, 'retry', () => this.retry())); break;
      case 'retrying': p(T.deepRetrying); break;
      case 'loading': break;
      default: for (const line of st.model?.lines ?? []) p(line);
    }
    if (st.state === 'too_few' && ctx.seq === 0) p(deep('deepNuclei', { start: startName(ctx.startSource) }), 'qc-deeper-muted');
  }
  /** The result: headline, bar and legend, then the sections. */
  paintResult(m, r, ctx, motion) {
    const before = this.shownModel?.state === 'ready' ? this.shownModel : null;
    this.shownModel = m;
    setText(this.stepNode, m.stepLine); setTitle(this.stepNode, m.stepTip);
    this.notes(this.notesNode, m.notes);
    setText(this.basisNode, m.basis); this.paintBasis(r, ctx);
    const start = startName(ctx.startSource);
    this.notes(this.afterNode, [...(r.seq === 'start' && ctx.seq === 0 ? [{ text: deep('deepNucleiShort', { start }), tip: deep('deepNuclei', { start }) }] : []), ...m.after]);
    this.bar.setAttribute('aria-label', m.barLabel);
    m.bar.texts && ['own', 'nb', 'loose'].forEach((k, i) => setText(this.legendItems[i], m.bar.texts[k]));
    // From neighbours (the same-kind count is the Neighbours label's tooltip).
    setHidden(this.noneNode, !m.none); setText(this.noneNode, m.noneText ?? '');
    const show = !m.none;
    setHidden(this.foreignHeading, !show || !m.chipsTip); setTitle(this.foreignHeading, m.chipsTip ?? '');
    setHidden(this.foreignNone, !show || !m.foreignNone); setText(this.foreignNone, m.foreignNone ?? '');
    setHidden(this.sameLine, !show || !m.sameLine); setText(this.sameLine, m.sameLine?.text ?? ''); setTitle(this.sameLine, m.sameLine?.tip ?? '');
    const enter = { chips: [], leave: [], rows: [] };
    this.paintChips(show ? m : null, motion, enter);
    const rows = this.paintSources(show ? m : null, motion, enter, r.iter);
    this.whereSync(r, m.none, rows, ctx);   // Show on the image: offered for these rows; while on, the glow and dots follow
    // Its own (the kind it looks like; its top genes here in that line's tooltip, or on a line of their own
    // when no kind is named), loose RNA, the table and the footer.
    const g = this.o.group?.(this.cell);
    setHidden(this.ownKind, !m.ownKind); setText(this.ownKind, m.ownKind ?? '');
    setTitle(this.ownKind, [T.deepOwnKindTooltip, m.ownTop, ...(g?.label ? [deep('deepOwnGroup', { group: g.label }), T.deepGroupsNote] : [])].join(' '));
    setHidden(this.ownGenes, !!m.ownKind); setText(this.ownGenes, m.ownGenes.text); setTitle(this.ownGenes, m.ownGenes.tip);
    setHidden(this.looseSection, !m.loose); setText(this.looseLine, m.loose?.text ?? ''); setTitle(this.looseLine, m.loose?.tip ?? '');
    setText(this.tableSummary, m.tableHeading);
    setHidden(this.tableRest, !m.tableRest); setText(this.tableRest, m.tableRest ?? '');
    this.tableVersion = (this.tableVersion ?? 0) + 1;
    if (this.tableBox.open) this.renderTable(false);   // not under keyboard focus: it follows once focus leaves
    this.paintHow(ctx);
    // The headline, the bars and the small line move; the rest lands with this paint.
    const to = { number: m.nb, widths: m.bar.widths };
    const from = motion && before ? { number: before.nb, widths: before.bar.widths } : to;
    this.sparkModel = this.sparkData(r);
    this.startMotion({ from, to, enter, motion, spark: motion && this.sparkModel ? this.sparkModel : null, number: m.p });
    this.focusModel(m, rows, motion, true);
  }
  lines(node, texts) {
    const key = JSON.stringify(texts);
    if (node._key === key) return;
    node._key = key; node.replaceChildren(...texts.map(t => this.el('p', '', t)));
    node.hidden = !texts.length;
  }
  /** Short notes ({text, tip}), each explained in its tooltip; rebuilt only when they change. */
  notes(node, items) {
    const key = JSON.stringify(items);
    if (node._key === key) return;
    node._key = key; node.replaceChildren(...items.map(n => { const p = this.el('p', '', n.text); if (n.tip) p.title = n.tip; return p; }));
    node.hidden = !items.length;
  }
  /** The basis's tooltip for the result shown (basisTooltip): what proseg reads as background in the result's own frame
   *  (the count beside it is that frame's: live it trails the frame on screen, looking back it may be the counted frame
   *  the engine snapped to), from that frame's cell statistics with its counted file (context cellStats(label, seq)),
   *  once they are in; else no number. */
  paintBasis(r = this.shown?.result ?? null, ctx = this.ctx()) {
    if (!this.isOpen || r?.state !== 'ready') return;
    const label = Number.isInteger(r.cell) ? r.cell + 1 : this.cell;   // the result's own cell (another may be on its way)
    let stats = null;
    try { stats = r.seq === 'start' ? null : ctx.cellStats?.(label, r.seq) ?? null; } catch { stats = null; }
    setTitle(this.basisNode, basisTooltip(r, stats));
  }
  paintHow(ctx) {
    const model = ctx.qc?.sampleModel ?? null, key = JSON.stringify(model ? [model.cells, model.created, model.seconds] : null);
    if (this.howList._key === key) return;
    this.howList._key = key; this.howList.replaceChildren(...howItems(model).map(t => this.el('li', '', t)));
  }
  /** The tagged genes as chips (at most CHIPS, then "+k more"), keyed by gene and ranked with hysteresis; a new chip
   *  slides in, a leaving one fades. Not rebuilt under a press or keyboard focus. */
  paintChips(m, motion, enter) {
    const chips = m?.chips ?? [];
    if ((this.pressed || this.focused === this.chips) && this.chipNodes) { this.stale = true; this.updateChipTexts(chips); return; }
    this.chipNodes ??= new Map();
    const byKey = new Map(chips.map(c => [c.key, c]));
    const order = rankOrder(this.order.chips, [...byKey.keys()], k => byKey.get(k).nb);
    this.order.chips = order;
    const visible = this.chipsAll ? order : order.slice(0, CHIPS), keep = new Set(visible);
    for (const [k, node] of this.chipNodes) if (!keep.has(k)) {
      this.chipNodes.delete(k);
      if (motion) { node.classList?.add('qc-deeper-leaving'); enter.leave.push(node); } else node.remove?.();
    }
    const nodes = visible.map(k => {
      let node = this.chipNodes.get(k);
      if (!node) {
        node = this.el('span', 'qc-deeper-chip'); node.dataset.gene = k; node.title = T.deepTagForeignTooltip;
        this.chipNodes.set(k, node);
        if (motion) enter.chips.push(node);
      }
      setText(node, byKey.get(k).text);
      return node;
    });
    const more = order.length - visible.length;
    setText(this.moreChip, more > 0 ? deep('deepForeignMore', { k: number(more) }) : '');
    this.chips.replaceChildren(...nodes, ...enter.leave, ...(more > 0 ? [this.moreChip] : []));
    setHidden(this.chips, !chips.length);
    this.paintWhereChips();
  }
  updateChipTexts(chips) { for (const c of chips) { const node = this.chipNodes?.get(c.key); if (node) setText(node, c.text); } }
  /** The source rows: neighbours holding >= 1 % of the cell's RNA, at most ROWS, by share with hysteresis; the rest fold
   *  into one line. Rows are keyed by cell and slide when their rank changes. */
  paintSources(m, motion, enter, iter = null) {
    if (!m) { setHidden(this.sourcesBlock, true); this.rows = []; return []; }
    setHidden(this.sourcesBlock, false);
    const { rows, fold, partly } = sourceRows(m, this.order.rows);
    this.rows = rows;
    const busy = this.pressed || this.focused === this.sources;
    if (busy && this.rowNodes) {
      this.stale = true;
      for (const row of rows) { const node = this.rowNodes.get(row.key); if (node) this.fillRow(node, row); }
    } else {
      this.stale = false;
      const before = this.order.rows;
      this.order.rows = rows.map(r => r.key);
      this.rowNodes ??= new Map();
      const keep = new Set(this.order.rows);
      for (const [k, node] of this.rowNodes) if (!keep.has(k)) { node.remove?.(); this.rowNodes.delete(k); }
      const nodes = rows.map((row, i) => {
        let node = this.rowNodes.get(row.key);
        if (!node) { node = this.makeRow(row); this.rowNodes.set(row.key, node); }
        this.fillRow(node, row);
        const was = before.indexOf(row.key);
        if (motion && was >= 0 && was !== i) enter.rows.push({ node, dy: (was - i) * ROW_PITCH });
        return node;
      });
      this.sources.replaceChildren(...nodes);
    }
    setHidden(this.sources, !rows.length);
    setHidden(this.spreadSources, rows.length > 0);
    setHidden(this.foldNode, !fold); setText(this.foldNode, fold?.text ?? '');
    // The rows' heading's tooltip: a rough guide, learned or not, the rows the model never saw as pairs, and how
    // many of the nearest neighbours look the same kind (no longer a line of its own).
    setTitle(this.sourcesHeading, [T.deepSourcesGuide, m.learnedNote, partly, m.sameCount].filter(Boolean).join(' '));
    // Groups from another iteration than the result's: named only when they are far from it or newer (groupsLag); the
    // usual lag of a pass or two goes to the Log.
    const g = rows.map(r => r.group).find(Boolean);
    const lag = g && groupsLag(g.iter, iter);
    const groupsFrom = lag ? deep('deepGroupsFrom', { i: number(g.iter) }) : null;
    if (g && finite(g.iter) && finite(iter) && g.iter !== iter && !lag && this.groupsNoted !== `${g.iter}:${iter}`) {
      this.groupsNoted = `${g.iter}:${iter}`; this.log(`cell groups from iteration ${g.iter} beside the result of iteration ${iter}`);
    }
    setHidden(this.groupsFrom, !groupsFrom); setText(this.groupsFrom, groupsFrom ?? '');
    if (this.rowNoteFor && !rows.some(r => r.label === this.rowNoteFor && r.same)) { this.rowNoteFor = 0; setHidden(this.rowNote, true); }
    this.paintStageNote();
    return rows;
  }
  /** A row at a fixed height (ROW_PITCH): its group's dot, its name, its share with a small bar, Show and ›; under them,
   *  muted, the genes it seems to give and its distance. Its words in full sit in its tooltip. */
  makeRow(row) {
    const li = this.el('li', 'qc-deeper-source'); li.dataset.cell = row.name;
    const main = this.el('div', 'qc-deeper-source-main');
    const sw = this.el('i', 'qc-deeper-group'), name = this.el('span', 'qc-deeper-source-name');
    const track = this.el('span', 'qc-deeper-source-bar'), fill = this.el('i'), share = this.el('span', 'qc-deeper-source-share');
    track.append(fill);
    const showB = this.button(T.deepSourceShowShort, 'show', () => this.showRow(li._label), 'qc-button qc-deeper-show');
    const deeper = this.button('', 'deeper', () => this.goNeighbour(li._label), 'qc-button qc-deeper-go');
    deeper.append(this.icon('arrow')); deeper.title = T.deepSourceDeeper; deeper.setAttribute('aria-label', T.deepSourceDeeper);
    main.append(sw, name, track, share, showB, deeper);
    const sub = this.el('div', 'qc-deeper-source-sub');
    li.append(main, sub);
    li._parts = { sw, name, fill, share, sub, showB };
    li.addEventListener?.('pointerenter', () => this.hover(li._label, 'row'));
    li.addEventListener?.('pointerleave', () => { if (this.hot === li._label) this.hover(0, 'row'); });
    return li;
  }
  fillRow(li, row) {
    const p = li._parts; li._label = row.label; li._same = !!row.same; li._row = row; li.dataset.cell = row.name; li.dataset.same = row.same ? '1' : '0';
    this.rowTitle(li);
    const color = row.color ?? COLORS.same;
    if (p.sw.dataset.color !== color) { p.sw.dataset.color = color; p.sw.style.background = color; }
    p.sw.title = row.group ? T.deepGroupsNote : T.deepGroupUnknown;
    setText(p.name, row.name);
    const width = `${(clamp(row.share, 0, 1) * 100).toFixed(1)}%`;
    if (p.fill.style.width !== width) p.fill.style.width = width;
    setText(p.share, row.pctText); setText(p.sub, row.sub);
    this.paintShow(p.showB, row.label, row.same);
    p.showB.disabled = !row.label;
    li.classList?.toggle('qc-deeper-lifted', !!row.label && this.hot === row.label);
  }
  /** A row's tooltip: its own words, and, while the glow lies near none of it (farLabel), that its pink dots more
   *  likely come from a cell of its kind with no outline here. */
  rowTitle(li) {
    const row = li?._row;
    if (!row) return;
    setTitle(li, row.label && row.label === this.farLabel() ? `${row.tooltip} ${deep('deepSourceNoOutlineTooltip', { cell: row.name })}` : row.tooltip);
  }
  paintStageNote() {
    const text = this.stageNote === '3d' ? T.deep3d : this.stageNote === 'zoom' ? T.deepZoom : '';
    setHidden(this.stageNoteNode, !text); setText(this.stageNoteNode, text);
  }
  /** What the overlay can show of the arrows now ('' drawn or none to draw, 'zoom', '3d'): main.js from DeeperArrows. */
  setStageNote(kind) {
    kind = kind === '3d' || kind === 'zoom' ? kind : '';
    if (kind === this.stageNote) return;
    this.stageNote = kind;
    if (this.isOpen && this.shownModel?.state === 'ready') this.paintStageNote();
  }
  /** The full table, virtualised: rows of TABLE_ROW_PX, TABLE_ROWS in view, the rest scroll inside. */
  renderTable(force = false) {
    const m = this.shownModel;
    if (!this.tableBox.open || m?.state !== 'ready') return;
    if (this.focused === this.tableBox && !force) { this.stale = true; return; }
    const rows = m.table, n = rows.length, top = this.tableScroll.scrollTop || 0;
    const first = clamp(Math.floor(top / TABLE_ROW_PX) - TABLE_PAD, 0, Math.max(0, n - 1)), last = Math.min(n, first + TABLE_ROWS + 2 * TABLE_PAD);
    const key = `${this.tableVersion}:${n}:${first}:${last}:${this.tableActive ?? ''}`;
    if (!force && key === this.tableKey_) return;
    this.tableKey_ = key;
    const spacer = h => { const tr = this.el('tr', 'qc-deeper-spacer'); const td = this.el('td'); td.colSpan = 4; tr.append(td); tr.style.height = `${h}px`; return tr; };
    const out = [];
    if (first > 0) out.push(spacer(first * TABLE_ROW_PX));
    for (let i = first; i < last; i++) out.push(this.tableRow(rows[i], i));
    if (last < n) out.push(spacer((n - last) * TABLE_ROW_PX));
    this.tbody.replaceChildren(...out);
  }
  /** A gene's row's tooltip: its own words, but, while the glow lies near none of the neighbour it names as giving most
   *  of the gene (farLabel), without that name and with what that neighbour's own row says (rowTitle). */
  tableTitle(g) {
    return g.giver?.label && g.giver.label === this.farLabel() ? `${g.unnamed} ${deep('deepSourceNoOutlineTooltip', { cell: g.giver.name })}` : g.tooltip;
  }
  tableRow(g, i) {
    const tr = this.el('tr'); tr.dataset.gene = g.gene; tr.dataset.foreign = g.flag ? '1' : '0'; tr.dataset.index = String(i);
    tr._row = g; tr.title = this.tableTitle(g); tr.tabIndex = (this.tableActive ?? 0) === i ? 0 : -1;
    if (g.flag) tr.className = 'qc-deeper-tagged';
    const name = this.el('td', 'qc-deeper-gene'); name.append(this.el('span', '', g.gene));
    if (g.flag) { const tag = this.el('small', 'qc-deeper-tag', T.deepTagForeign); tag.title = T.deepTagForeignTooltip; name.append(tag); }
    const tx = this.el('td', 'qc-deeper-num', count(g.tx));
    const split = this.el('td'), bar = this.el('span', 'qc-deeper-row-bar');
    const total = g.tx || 1;
    for (const [k, v] of [['own', g.own], ['nb', g.nb], ['loose', g.loose]]) { const s = this.el('i', `qc-deeper-part-${k}`); s.style.width = `${(clamp(v / total, 0, 1) * 100).toFixed(1)}%`; bar.append(s); }
    split.append(bar);
    tr.append(name, tx, split, this.el('td', 'qc-deeper-num', deep('deepRowPercent', { p: g.pct })));
    return tr;
  }
  tableKey(delta) {
    const m = this.shownModel; if (m?.state !== 'ready' || !m.table.length) return;
    const i = clamp((this.tableActive ?? 0) + delta, 0, m.table.length - 1);
    this.tableActive = i;
    const top = i * TABLE_ROW_PX, view = TABLE_ROWS * TABLE_ROW_PX, st = this.tableScroll.scrollTop || 0;
    if (top < st) this.tableScroll.scrollTop = top; else if (top + TABLE_ROW_PX > st + view) this.tableScroll.scrollTop = top + TABLE_ROW_PX - view;
    this.renderTable(true);
    const row = [...(this.tbody.children || [])].find(tr => tr.dataset?.index === String(i));
    try { row?.focus?.({ preventScroll: true }); } catch { /* focus is a convenience */ }
  }
  rerender() { this.stale = false; this.shownKey = null; this.update(true); }

  // ---- motion -------------------------------------------------------------------------------------------------------------
  startMotion(plan) {
    this.stopMotion(true);
    const { from, to, enter, motion } = plan;
    const glides = (a, b) => a.some((w, i) => Math.abs(w - b[i]) > 1e-4);
    const moves = motion && (from.number !== to.number || glides(from.widths, to.widths)
      || enter.chips.length || enter.leave.length || enter.rows.length || plan.spark);
    if (!moves) { this.paintFinal(plan); return; }
    this.motion = { ...plan, start: this.now(), step: 0 };
    for (const node of enter.chips) node.classList?.add('qc-deeper-pulse');
    this.paintStep(0);
    this.nextStep();
  }
  nextStep() {
    const m = this.motion; if (!m) return;
    const due = m.start + (m.step + 1) * STEP_MS;
    this.later('motion', due - this.now(), () => {
      this.motionFrame = this.clock.requestAnimationFrame(() => {
        this.motionFrame = null;
        if (this.motion !== m) return;
        m.step++; this.paintStep(m.step * STEP_MS);
        this.harness?.count('qcDeeperAnimFrames');
        if (m.step >= MAX_PAINTS - 1) this.stopMotion(true); else this.nextStep();
      });
    });
  }
  paintStep(elapsed) {
    const m = this.motion; if (!m) return;
    const count_ = motionProgress(elapsed, NUMBER_MS), glide = motionProgress(elapsed, GLIDE_MS);
    const value = Math.round(m.from.number + (m.to.number - m.from.number) * count_);
    setText(this.numberNode, deep('deepNumber', { p: count_ >= 1 || value === m.to.number ? m.number : number(value) }));
    this.paintWidths(m.from.widths.map((w, i) => w + (m.to.widths[i] - w) * glide));
    for (const node of m.enter.chips) { node.style.opacity = String(glide.toFixed(3)); node.style.transform = glide >= 1 ? '' : `translateX(${((1 - glide) * -10).toFixed(1)}px)`; }
    for (const node of m.enter.leave) node.style.opacity = String((1 - glide).toFixed(3));
    for (const { node, dy } of m.enter.rows) node.style.transform = glide >= 1 ? '' : `translateY(${(dy * (1 - glide)).toFixed(1)}px)`;
    if (m.spark) this.drawSpark(m.spark, glide);
  }
  /** The end state of an update, in one paint (also what a stopped motion jumps to). */
  paintFinal(plan) {
    setText(this.numberNode, deep('deepNumber', { p: plan.number }));
    this.paintWidths(plan.to.widths);
    for (const node of plan.enter?.chips ?? []) { node.style.opacity = ''; node.style.transform = ''; node.classList?.remove('qc-deeper-pulse'); }
    for (const node of plan.enter?.leave ?? []) node.remove?.();
    for (const { node } of plan.enter?.rows ?? []) node.style.transform = '';
    if (this.sparkModel) this.drawSpark(this.sparkModel, 1); else setHidden(this.spark, true);
  }
  stopMotion(finish = false) {
    this.cancel('motion');
    if (this.motionFrame != null) { this.clock.cancelAnimationFrame(this.motionFrame); this.motionFrame = null; }
    const m = this.motion; this.motion = null;
    if (finish && m) this.paintFinal(m);
  }
  paintWidths(widths, segments = this.segments) {
    widths.forEach((w, i) => {
      const seg = segments[i], grow = (Math.max(0, w) * 1000).toFixed(1), min = w > 0 ? '2px' : '0px';
      if (seg.style.flexGrow !== grow) seg.style.flexGrow = grow;
      if (seg.style.minWidth !== min) seg.style.minWidth = min;
    });
  }

  // ---- the small line -----------------------------------------------------------------------------------------------------
  /** The small line for result `r`: its cell's points up to r's iteration, r's own byte at it (whatever the history read
   *  there: the answer shown is the one drawn). */
  sparkData(r) {
    const h = this.histories.get(this.cell - 1), own = r.state === 'ready' && r.seq !== 'start' && finite(r.iter);
    const points = (h?.points ?? []).filter(([i, c]) => finite(i) && Number.isInteger(c) && c < 255 && (r.seq === 'start' || !finite(r.iter) || i <= r.iter)
      && !(own && i === r.iter));
    if (own) points.push([r.iter, r.code]);
    points.sort((a, b) => a[0] - b[0]);
    const start = Number.isInteger(h?.start) && h.start < 255 ? h.start : null;
    if (points.length < 2 && start == null) return null;
    return { points, start, end: Math.max(1, ...points.map(p => p[0]), (this.ctx().frames || []).at(-1)?.iter ?? 1) };
  }
  drawSpark(s, t = 1) {
    setHidden(this.spark, false);
    setText(this.sparkCaption, s.start != null ? deep('deepSparkStart', { q: number(Math.floor(cellShare(s.start) * 100 + 1e-9)) }) : '');
    const canvas = this.sparkCanvas, ratio = globalThis.devicePixelRatio || 1, W = 96, H = 28;
    if (canvas.width !== Math.round(W * ratio)) { canvas.width = Math.round(W * ratio); canvas.height = Math.round(H * ratio); }
    const g = canvas.getContext?.('2d'); if (!g) return;
    g.setTransform?.(ratio, 0, 0, ratio, 0, 0); g.clearRect?.(0, 0, W, H);
    const x = i => 2 + (W - 6) * clamp(i / s.end, 0, 1), y = c => H - 3 - (H - 6) * clamp(cellShare(c), 0, 1);
    if (s.start != null) { g.strokeStyle = 'rgba(154,164,178,.55)'; g.lineWidth = 1; g.setLineDash?.([2, 2]); g.beginPath(); g.moveTo(2, y(s.start)); g.lineTo(W - 2, y(s.start)); g.stroke(); g.setLineDash?.([]); }
    const pts = s.points; if (!pts.length) return;
    g.strokeStyle = COLORS.nb; g.lineWidth = 1.5; g.beginPath();
    const n = pts.length, lastFull = t >= 1 ? n : n - 1;
    for (let k = 0; k < lastFull; k++) { const [i, c] = pts[k]; if (k) g.lineTo(x(i), y(c)); else g.moveTo(x(i), y(c)); }
    let end = pts[n - 1];
    if (t < 1 && n >= 2) {
      const [i0, c0] = pts[n - 2], [i1, c1] = pts[n - 1], xe = x(i0) + (x(i1) - x(i0)) * t, ye = y(c0) + (y(c1) - y(c0)) * t;
      g.lineTo(xe, ye); end = null; g.stroke(); g.fillStyle = '#fff'; g.beginPath(); g.arc?.(xe, ye, 2, 0, Math.PI * 2); g.fill?.(); return;
    }
    g.stroke();
    if (end) { g.fillStyle = '#fff'; g.beginPath(); g.arc?.(x(end[0]), y(end[1]), 2, 0, Math.PI * 2); g.fill?.(); }
  }
  sparkHover(e) {
    const s = this.sparkModel; if (!s?.points.length) return;
    const xCss = finite(e.offsetX) ? e.offsetX : 0, i = clamp(Math.round((xCss - 2) / 90 * s.end), 0, s.end);
    let best = s.points[0];
    for (const p of s.points) if (Math.abs(p[0] - i) < Math.abs(best[0] - i)) best = p;
    const text = deep('deepSparkHover', { i: number(best[0]), p: number(Math.floor(cellShare(best[1]) * 100 + 1e-9)) });
    if (this.sparkCanvas.title !== text) this.sparkCanvas.title = text;
  }

  // ---- the stage: outlines, greying out, arrows, rings ---------------------------------------------------------------------
  /** What the stage, the overlay and the UMAP show for the window (focus): the window's cell; the sources (rows that look
   *  like a different kind; their share 0-1); the neighbours that look like its own kind; the row hovered or pinned; the
   *  cells that stay lit while every other one greys out (the cell and its shown rows, same-kind ones included,
   *  the set Show on the image lights; at most LIT_MAX) and how far (dim, the stage eases it in and out). `apply`: the
   *  focus of a result the window paints (render), whose arrows the page may draw with its next paint batch; any other
   *  change of it (a row hovered or pinned, the glow's far row) is drawn at once. */
  focusModel(m, rows = [], motion = false, apply = false) {
    if (!m || m.state !== 'ready' || !this.isOpen) { this.setFocus(null); return; }
    // The row none of the glow lies near (farLabel) is not where the pink dots come from: neither outlined, ringed nor
    // lit while the view says so.
    const far = this.farLabel();
    const sources = m.none ? [] : rows.filter(r => r.label && !r.same && r.label !== far).map(r => ({ label: r.label, share: r.share, pct: r.pct, name: r.name, group: r.groupText }));
    const same = m.none ? [] : m.neighbours.filter(n => n.same && n.label).map(n => n.label);   // under 5 %: no neighbour is listed
    const lit = [...new Set([this.cell, ...(m.none ? [] : rows.map(r => r.label).filter(l => l > 0 && l !== far))])].slice(0, LIT_MAX);
    this.setFocus({ cell: this.cell, sources, same, hot: this.hot || this.pin, lit, dim: 1, motion, apply });
  }
  setFocus(focus) {
    const key = focus ? JSON.stringify([focus.cell, focus.sources.map(s => [s.label, Math.round(s.share * 1000)]), focus.same, focus.hot,
      focus.lit, focus.dim]) : null;
    if (key === this.focusKey) return;
    this.focusKey = key; this.focus = focus;
    this.call('onFocus', focus);
  }
  /** A row's Show: only the transcripts that likely come from that neighbour glow (Show on the image turned
   *  on if it was off), the camera fits the cell and that neighbour, and its highlight stays pinned; pressed again,
   *  every suspect glows again. A neighbour of the same kind explains instead (no gene tells its RNA apart). */
  showRow(label) {
    if (!label) return;
    const row = this.rows.find(r => r.label === label);
    if (row?.same) { this.rowNoteFor = this.rowNoteFor === label ? 0 : label; setHidden(this.rowNote, !this.rowNoteFor); return; }
    if (this.rowNoteFor) { this.rowNoteFor = 0; setHidden(this.rowNote, true); }
    const on = this.pin !== label;
    this.pin = on ? label : 0;
    this.hot = this.pin || this.hot;
    this.repaintRows(); this.refocus();
    if (!on) { if (this.where) { this.where.row = 0; this.paintGlow(); this.paintWhere(); } return; }
    if (!this.where) { this.toggleWhere(label); return; }
    this.where.row = label; this.paintGlow(); this.paintWhere(); this.flyTo();
  }
  goNeighbour(label) { if (!label) return; this.log(`went to the neighbour ${this.nameOf(label)}`); this.target(label, 'neighbour'); this.call('onSelect', label); }
  /** A source cell or its arrow under the pointer on the stage, or a row under the pointer (lifts the row, lights the
   *  arrow and outline; a pinned neighbour stays lit). */
  hover(label, from = 'stage') {
    label = label > 0 ? label : 0;
    const sources = new Set((this.focus?.sources ?? []).map(s => s.label));
    const next = sources.has(label) ? label : 0;
    if (next === this.hot) return;
    this.hot = next;
    this.repaintRows(); this.refocus();
  }
  repaintRows() {
    for (const node of this.rowNodes?.values() ?? []) {
      node.classList?.toggle('qc-deeper-lifted', !!node._label && (this.hot === node._label || this.pin === node._label));
      if (node._parts?.showB) this.paintShow(node._parts.showB, node._label, node._same);
    }
  }
  /** A row's Show: pressed while that neighbour's suspects glow; a same-kind row's says why nothing would. */
  paintShow(button, label, same = false) {
    const on = !same && !!label && this.pin === label;
    setText(button, on ? T.deepSourceShownShort : T.deepSourceShowShort);
    const tip = same ? T.deepShowSameKind : on ? T.deepSourceShown : T.deepSourceShow;
    if (button.title !== tip) { button.title = tip; button.setAttribute('aria-label', tip); }
    button.setAttribute('aria-pressed', String(on));
  }
  refocus() { if (this.focus) this.setFocus({ ...this.focus, hot: this.hot || this.pin, motion: false, apply: false }); }
  /** The stage tooltip's extra line for a cell under the pointer: a source ("about p % of {target}'s RNA seems to come
   *  from here"), or a neighbour that looks like the same kind; null otherwise. */
  tooltipLine(label) {
    if (!this.isOpen || !(label > 0) || !this.focus || label === this.cell) return null;
    const target = this.nameOf(this.cell);
    const s = this.focus.sources.find(x => x.label === label);
    if (s) return deep('deepArrowTooltip', { cell: s.name, group: s.group, p: number(Math.max(1, finite(s.pct) ? s.pct : partPercent(s.share))), target });
    if (this.focus.same.includes(label)) return deep('deepSameKindTooltip', { cell: this.nameOf(label), target });
    return null;
  }

  // ---- where it spills: Show on the image, the glow, Where they sit --------------------------------------------------------
  /** Show on the image and "Test a setting here…" in their row: the toggle while a ready result has source rows to show
   *  (whereWanted), the test beside it while the page can test this cell (context testable: never greyed out; its form
   *  lists what refuses a test, and Run's tooltip what a press waits for first, the run's end or the check's included).
   *  While the window says the check is off, the test sits beside the check's switch instead (the off row; a press of
   *  its Run switches the check on first: check_off, a wait in Run's tooltip). Neither in small mode (CSS). The one entry
   *  moves between the two rows, so the harness's button[data-action=test-open] is one button. */
  paintWhereRow() {
    const show = this.isOpen && !!this.whereWanted, off = this.isOpen && !show && this.root.dataset.state === 'off';
    const row = off ? this.offRow : this.whereRow;
    if (this.testButton.parentElement !== row) row.append(this.testButton);   // last: after the toggle, or the switch
    setHidden(this.whereButton, !show);
    setHidden(this.testButton, !(show || off) || !this.ctx().testable);
    setHidden(this.whereRow, !show);
  }
  /** "Test a setting here…" pressed: the mixing check's test of a setting on this cell (onTest(label, name)). */
  openTest() {
    if (!this.isOpen || !this.cell) return;
    this.log(`test a setting from Go deeper: ${this.nameOf(this.cell)}`);
    this.call('onTest', this.cell, this.nameOf(this.cell));
  }
  /** The result just painted (`r`, its source rows `rows`): the toggle is offered for a ready result with source rows;
   *  while it is on, the glow and the dots follow the result shown (none to show any more: off). */
  whereSync(r, none, rows, ctx = this.ctx()) {
    const req = this.whereWant(r, none, rows);
    this.whereWanted = req;
    this.paintWhereRow();
    const w = this.where;
    if (!w) return;
    if (!req) { this.whereOff('no source rows'); return; }
    // The window's result for the reply's own frame replaced (another model, or another answer for that frame): the
    // reply's suspects scored again on it, so the glow, its line and the suspects a test starts with follow the result
    // shown.
    const held = w.reply ? this.results.get(w.reply.cell)?.get(w.reply.seq) : null;
    if (held && held !== w.result) { this.whereScore(held); this.paintGlow(); this.paintWhere(); }
    else if (w.reply && RUN_ENDED.has(ctx.run?.state) !== this.whereEnded) this.paintWhere();   // the run ended: its tooltips say so
    this.whereRest(req, ctx);
    this.whereAskSoon(req, ctx);
  }
  /** Not following the run as it plays (paused, stepped or scrubbed), the window's request `req` is answered for its own
   *  frame: an exact answer kept for it from the counted frame before (w.snapped, only while live) is let go, and once
   *  the run rests on that frame (proseg acknowledged the pause there) an answer that is not exact is asked again, once
   *  per request (one asked at rest already counts), as the engine then gives proseg's exact share of it. True when it
   *  is to be asked again. */
  whereRest(req, ctx = this.ctx()) {
    const w = this.where, ack = ctx.run?.ack;
    if (!w || !req || ctx.live) return false;
    const rest = ctx.run?.state === 'paused' && ack?.state === 'paused' && ack.seq === req.params.seq;
    const again = !!w.snapped || (rest && !!w.reply && w.key === req.key && w.reply.held !== 'exact' && w.restKey !== req.key);
    if (!again) return false;
    if (rest) w.restKey = req.key;
    Object.assign(w, { key: null, snapped: null });
    this.paintWhere();
    return true;
  }
  /** What the window asks for result `r` (whereRequest), its key naming the result's count as well: an answer for the
   *  same frame on the other count (the frame counted again for the hold, a finish or the end) asks again. */
  whereWant(r, none, rows) {
    const req = whereRequest(r, { cell: this.cell, none, rows, chips: this.order.chips });
    return req && { ...req, key: JSON.stringify([req.key, r.count ?? 'averaged']) };
  }
  /** The toggle pressed: on (the shown result's dots asked at once; `row`: a row's Show turned it on, its neighbour's
   *  suspects glow) or off. */
  toggleWhere(row = 0) {
    if (this.where) { this.whereOff('pressed'); return; }
    const req = this.whereWanted;
    if (!req || !this.isOpen) return;
    this.where = { slots: new Map(), ownSlots: new Map(), wanted: req, key: null, asking: null, askedAt: -Infinity, req: null, reply: null,
      error: null, waits: 0, tries: 0, flown: false, working: false, shown: null, colors: [], counts: {}, row, result: null, scores: null, scored: 0,
      glow: null, lit: null, startKey: null, startParams: null, start: null, places: null, holds: null, snapped: null, restKey: null };
    this.harness?.count('qcDeeperWhereOn');
    this.log(`dots on the image for ${this.nameOf(this.cell)}`);
    this.sendStartOutline(true);
    this.paintWhere();
    this.whereAskSoon(req);
  }
  /** Ask for `req` unless its answer is held or on its way: one request at a time (the newest wanted follows the answer),
   *  at most once a second while live; at once paused, stepped or scrubbed (a frame asked once). The start line's reply
   *  for the same cell and genes goes with it, once per set (whereStart). */
  whereAskSoon(req, ctx = this.ctx()) {
    const w = this.where;
    if (!w || !req) return;
    w.wanted = req;
    this.whereStart(req, ctx);
    if (w.key === req.key || w.asking || this.timers.has('where-ask')) return;
    const wait = ctx.live ? WHERE_REFRESH_MS - (this.now() - w.askedAt) : 0;
    if (wait > 0) { this.later('where-ask', wait, () => this.whereAskSoon(this.where?.wanted)); return; }
    this.whereAsk(req);
  }
  async whereAsk(req) {
    const w = this.where, run = this.runId;
    if (!w || !run || typeof this.api.qcDeeperWhere !== 'function') return;
    w.asking = req.key; w.askedAt = this.now();
    this.harness?.count('qcDeeperWhereAsked');
    if (!w.reply) this.later('where-working', WORKING_MS, () => { w.working = true; this.paintWhere(); });   // "Finding them…" after 250 ms
    let raw = null, failure = null;
    const t0 = this.now();
    try { raw = await this.api.qcDeeperWhere(run, req.params); } catch (error) { failure = error ?? {}; }
    try { globalThis.performance?.measure?.('plv:qc-deeper-where-fetch', { start: t0, detail: { seq: req.params.seq } }); globalThis.performance?.clearMeasures?.('plv:qc-deeper-where-fetch'); }
    catch { /* a profiling aid only: the request to its parsed answer */ }
    if (this.where !== w || run !== this.runId) return;   // turned off (or another run) meanwhile: dropped
    w.asking = null;
    const reply = failure ? null : normaliseWhere(raw, req);
    if (!reply) { this.whereFail(req, failure ?? { status: 422 }); return; }
    this.whereTake(reply, req);
  }
  /** Where they sat at the start: the same cell, sources and genes asked at the start line (seq "start"; its own
   *  genes left out, they never glow; both lists sorted, so a new rank asks nothing again), once per set: its points tell
   *  where each glowing transcript sat then (startPlaces). None for a run from nuclei only (no starting cells), for the
   *  start line's own view, or without a neighbour's gene to glow. A refusal leaves the line without its places; a lost
   *  connection is asked again with the next result. */
  async whereStart(req, ctx = this.ctx()) {
    const w = this.where, run = this.runId;
    if (!w || !run || !req?.params.genes.length || req.params.seq === 'start' || ctx.startSource === 'nuclei'
      || typeof this.api.qcDeeperWhere !== 'function') return;
    const params = { seq: 'start', cell: req.params.cell, sources: [...req.params.sources].sort((a, b) => a - b), genes: [...req.params.genes].sort(), own: [] };
    const key = JSON.stringify([params.cell, params.sources, params.genes]);
    if (w.startKey === key) return;   // asked: answered, on its way or refused
    const had = w.places;
    Object.assign(w, { startKey: key, startParams: params, start: null, places: null });
    if (had) { this.paintGlow(); this.paintWhere(); }   // another set: what the old one said goes until its answer
    this.harness?.count('qcDeeperWhereStartAsked');
    let raw = null, failure = null;
    try { raw = await this.api.qcDeeperWhere(run, params); } catch (error) { failure = error ?? {}; }
    if (this.where !== w || run !== this.runId || w.startKey !== key) return;   // off, another run or another set meanwhile
    const start = failure ? null : normaliseWhere(raw, { params });
    if (!start) {
      if (failure && !Number.isInteger(failure.status)) w.startKey = null;
      this.log(`could not tell what ${this.nameOf(this.cell)} held at the start: ${failure?.status ? `HTTP ${failure.status}` : failure ? 'no connection' : 'not a reply'}`);
      return;
    }
    w.start = start; this.placeStart(w);
    this.paintGlow(); this.paintWhere();
  }
  /** Where the glowing transcripts of the reply shown sat at the start (startPlaces against the start line's reply for
   *  the same cell, neighbours and genes); null until that reply is in, or when it was asked for another set. */
  placeStart(w = this.where) {
    w.places = w.reply && w.start && w.req && w.startParams ? startPlaces(w.reply, w.start, w.req.params.sources, w.startParams.sources) : null;
  }
  /** How the start held each point of the reply shown (context startHold: a detail run's start layers), asked once per
   *  reply that places some inside its own starting outline; null where the page has none (asked again with the next
   *  paint while it says none: the start layers may come in later). */
  holdsOf(w = this.where) {
    if (!w?.reply || !w.places?.includes(1)) return null;
    if (w.holds?.reply === w.reply) return w.holds.value;
    let value = null;
    try { value = this.ctx().startHold?.(w.reply, w.reply.cell + 1) ?? null; } catch { value = null; }
    if (!(value?.length === w.reply.count)) return null;
    w.holds = { reply: w.reply, value };
    return value;
  }
  /** The cell's starting outline on the stage while the view is on (none for a run from nuclei only, which has
   *  no starting cells): onStartOutline({cell: label, name}) or null, each change once. */
  sendStartOutline(on) {
    const outline = on && this.cell && this.ctx().startSource !== 'nuclei' ? { cell: this.cell, name: this.nameOf(this.cell) } : null;
    const key = outline ? `${this.runId}:${outline.cell}` : null;
    if (key === this.startOutlineKey) return;
    this.startOutlineKey = key;
    this.call('onStartOutline', outline);
  }
  /** A request failed: a frame not saved yet (409) is asked again after WHERE_WAIT_MS (WHERE_WAITS times), a lost
   *  connection after 2, 5, 10 and 30 s; anything else is said and asked again only for another frame. The dots on the
   *  stage stay as they were. */
  whereFail(req, error) {
    const w = this.where, status = Number.isInteger(error?.status) ? error.status : null;
    this.cancel('where-working');
    if (status === 409 && w.waits < WHERE_WAITS) {
      w.waits++; w.error = { kind: 'waiting', status };
      this.later('where-ask', WHERE_WAIT_MS, () => this.whereAskSoon(this.where?.wanted));
    } else if (status == null && w.tries < RETRY_MS.length) {
      w.error = { kind: 'failed', status }; this.later('where-ask', RETRY_MS[w.tries++], () => this.whereAskSoon(this.where?.wanted));
    } else { w.error = { kind: 'failed', status }; w.key = req.key; }
    this.log(`could not show ${this.nameOf(this.cell)}'s dots for ${req.params.seq}: ${status ? `HTTP ${status}` : 'no connection'}`);
    this.paintWhere();
    if (w.wanted && w.wanted.key !== req.key) this.whereAskSoon(w.wanted);
  }
  /** An answer: the dots and the fading to the stage (Stage.setSpill), the suspects scored on the result the request was
   *  made for and their glow (Stage.setGlow), the camera's move (the first answer since the toggle went on), the chips'
   *  colours and Where they sit. Following a cell as the run plays on, an approximate answer that comes right after an
   *  exact one of the counted frame just before it for the same request (snappedBy) leaves the exact one on screen, said
   *  so (w.snapped, deepWhereSnapped), and is not asked again while the run plays (whereRest asks it once it stops); any
   *  exact answer, or one a counted frame further on, replaces it. */
  whereTake(reply, req) {
    const w = this.where, t0 = this.now();
    this.cancel('where-working');
    if (this.snappedBy(w.reply, reply, w.req, req)) {
      Object.assign(w, { key: req.key, error: null, waits: 0, tries: 0, snapped: { iter: w.reply.iter } });
      this.harness?.count('qcDeeperWhereSnapped');
      this.paintWhere();
      if (w.wanted && w.wanted.key !== w.key) this.whereAskSoon(w.wanted, this.ctx());
      return;
    }
    Object.assign(w, { key: req.key, req, reply, error: null, waits: 0, tries: 0, snapped: null });
    const from = reply.genes.filter((g, i) => reply.roles[i] === 'from'), own = reply.genes.filter((g, i) => reply.roles[i] === 'own');
    assignSlots(w.slots, from, WHERE_COLORS.from.length); assignSlots(w.ownSlots, own, WHERE_COLORS.own.length);
    w.colors = reply.genes.map((g, i) => (reply.roles[i] === 'own' ? WHERE_COLORS.own[w.ownSlots.get(g)] : WHERE_COLORS.from[w.slots.get(g)]) ?? COLORS.same);
    w.counts = Object.fromEntries(reply.genes.map(g => [g, 0]));
    for (let i = 0; i < reply.count; i++) w.counts[reply.genes[reply.gene[i]]]++;
    this.whereScore(this.results.get(reply.cell)?.get(reply.seq) ?? null);
    this.placeStart(w);   // where they sat at the start, once that reply is in
    w.shown = null; this.paintGlow(false);   // the glow first: the cells lit leave out the row it lies near none of
    w.shown = { key: `${this.runId}:${req.key}`, cell: this.cell, lit: this.litOf(w), points: wherePoints(reply, req.labels.length),
      colors: w.colors };
    this.call('onSpill', w.shown);
    this.relight(false);
    if (!w.flown && reply.box) { w.flown = true; this.flyTo(); }   // once, in 2-D and 3-D
    this.harness?.count('qcDeeperWhereShown');
    this.paintWhere(); this.paintWhereChips();
    try { globalThis.performance?.measure?.('plv:qc-deeper-where', { start: t0, detail: { seq: reply.seq, points: reply.count } }); globalThis.performance?.clearMeasures?.('plv:qc-deeper-where'); }
    catch { /* a profiling aid only */ }
    // A newer frame wanted meanwhile, or an answer worked out while the run still played that came back once it rests
    // on that frame (whereRest): asked again.
    const ctx = this.ctx();
    if (w.wanted && (w.wanted.key !== w.key || this.whereRest(w.wanted, ctx))) this.whereAskSoon(w.wanted, ctx);
  }
  /** Whether `reply` (approximate, the answer to `req`) is kept off the screen for `shown` (the answer to `shownReq`),
   *  the exact reply of the counted frame just before it, while the window follows the run as it plays: the same cell,
   *  rows and genes asked, the frame apart, and no frame strictly between them with proseg's count of its own (context
   *  counted), so the exact shares of the frame before stand for it rather than approximate ones. Never at rest, where
   *  the engine gives the frame's own exact shares (whereRest), nor for other rows (the old neighbour's dots would stay
   *  under the new one's name). */
  snappedBy(shown, reply, shownReq, req, ctx = this.ctx()) {
    if (!ctx.live || shown?.held !== 'exact' || reply?.held !== 'approx' || shown.cell !== reply.cell) return false;
    if (!Number.isInteger(shown.seq) || !Number.isInteger(reply.seq) || !(shown.seq < reply.seq)) return false;
    const asked = r => r && JSON.stringify([r.params.cell, r.params.sources, r.params.genes, r.params.own]);
    if (!asked(shownReq) || asked(shownReq) !== asked(req)) return false;
    const counted = ctx.counted;
    for (let s = shown.seq + 1; s < reply.seq; s++) {
      let yes = false;
      try { yes = counted?.(s) === true; } catch { yes = false; }
      if (yes) return false;
    }
    return true;
  }
  /** The reply's suspects scored on `result`, the split they are built on: the window's own result for the reply's
   *  frame (its model, never the engine's current one). The view is bound to it (w.result): its glow (a new key for
   *  each scoring), line and the suspects kept for a test of a setting all come from that scoring. */
  whereScore(result) {
    const w = this.where;
    w.result = result; w.scores = suspectScores(w.reply, result); w.scored++;
    this.keepSuspects(w.reply, w.scores);
  }
  /** The camera's move for the view: the cell and every row, or, for a row's Show, the cell and that neighbour; when none
   *  of the glow lies near the neighbour it would name, the cell and its glow (never that neighbour, which may lie far
   *  off). In 3-D the angle stays the user's: the stage fits the box itself, standing up to the slab's top, into the
   *  part of the stage the window leaves free (onFly's view {box, free}). */
  flyTo() {
    const w = this.where, ctx = this.ctx(), reply = w?.reply;
    if (!reply?.box) return;
    const k = w.row ? w.req.labels.indexOf(w.row) : -1;
    const box = (w.glow?.notNear != null ? glowBoxOf(reply, w.lit) : null) ?? (k >= 0 ? pairBox(reply, k) : null) ?? reply.box;
    this.call('onFly', this.whereRect(box, ctx), !this.reducedMotion, ctx.view3d ? { box: [...box], free: this.whereFree(box, ctx) } : null);
  }
  /** The window's place and the open test corner (context cornerBox), which the camera's move keeps clear of. */
  whereClear(ctx = this.ctx()) {
    let corner = null;
    try { corner = ctx.cornerBox?.() ?? null; } catch { corner = null; }
    return { win: { left: this.left ?? 0, top: this.top ?? 0, w: this.size.w || WIDTH, h: this.size.h || 0 }, corner };
  }
  /** The world rect the camera's move fits for `box` ([x0, y0, x1, y1] µm): whereView beside the window where it is and the
   *  open test corner (context cornerBox). The harness's fair parity takes the same rect. */
  whereRect(box, ctx = this.ctx()) {
    const { win, corner } = this.whereClear(ctx);
    return whereView(box, ctx.stage, win, corner);
  }
  /** The part of the stage (CSS px) the camera's move puts `box` in (whereFree, beside the window and clear of the corner). */
  whereFree(box, ctx = this.ctx()) {
    const { win, corner } = this.whereClear(ctx);
    return whereFree(box, ctx.stage, win, corner);
  }
  /** The glow for the view: every suspect, or only a row's (its Show); none to show: off. Its line's counts are kept for
   *  Where they sit and the snapshot (places: where those the line counts sat at the start, and start, its own, null
   *  until known; levels: the start's hold of those inside its own outline, null without the start layers). */
  paintGlow(relight = true) {
    const w = this.where;
    if (!w?.reply) { this.sendGlow(null); return; }
    const k = w.row ? w.req.labels.indexOf(w.row) : -1;
    const g = glowPoints(w.reply, w.scores, k >= 0 ? k : null, w.places, this.holdsOf(w));
    w.glow = { count: g.count, likely: g.likely, near: g.near, start: g.start, places: g.places, levels: g.levels, row: k >= 0 ? w.row : null,
      notNear: g.notNear };
    w.lit = g.points;   // the glowing points (the camera's box when none lies near the neighbour it would name)
    const giver = g.giver != null ? sourceCentre(w.reply, g.giver) : null;
    this.sendGlow(g.count ? { key: `${this.runId}:${w.key}:${w.result?.model ?? ''}:${w.scored}:${k}`, points: g.points, depth: g.depth, giver } : null);
    if (relight) this.relight();
  }
  /** The source row the glow lies near none of (A2: glowPoints' notNear), by its label, while the view shows a glow;
   *  else 0. That neighbour is not presented as where the pink dots come from. */
  farLabel() {
    const w = this.where, k = w?.glow?.count > 0 ? w.glow.notNear : null;
    return Number.isInteger(k) ? w.req?.labels?.[k] ?? 0 : 0;
  }
  /** The cells the view keeps lit (the stage's fading): the cell and its rows', but the row the glow lies near none of. */
  litOf(w = this.where) {
    const far = this.farLabel();
    return [this.cell, ...(w?.req?.labels ?? []).filter(l => l !== far)];
  }
  /** The row the glow lies near none of changed (farLabel): the rows' tooltips (the gene table's too), the cells lit
   *  (`spill`: handed to the stage again, unless the caller just did) and the window's focus (the stage's outlines and
   *  fading, the UMAP's rings) follow. */
  relight(spill = true) {
    const far = this.farLabel();
    if (far === (this.farShown ?? 0)) return;
    this.farShown = far;
    for (const node of this.rowNodes?.values() ?? []) this.rowTitle(node);
    for (const tr of this.tbody?.children ?? []) if (tr._row) tr.title = this.tableTitle(tr._row);
    const w = this.where;
    if (spill && w?.shown) { w.shown = { ...w.shown, lit: this.litOf(w) }; this.call('onSpill', w.shown); }
    if (this.shownModel?.state === 'ready') this.focusModel(this.shownModel, this.rows);
  }
  sendGlow(glow) {
    const key = glow?.key ?? null;
    if (key === this.glowKey) return;
    this.glowKey = key;
    this.call('onGlow', glow);
  }
  /** The suspects of the reply's cell, kept for a test of a setting (suspects()): P >= GLOW_MIN, the top SUSPECTS_SENT;
   *  one whose gene and place other points of the reply share also carries its `rank` among them by depth and how many
   *  they are (`of`: samePlace), so a test's runs can tell which transcript it is. */
  keepSuspects(reply, scores) {
    const list = [];
    for (let i = 0; i < reply.count; i++) {
      const P = scores.p[i];
      if (P >= GLOW_MIN) list.push({ i, p: +P.toFixed(3) });
    }
    list.sort((a, b) => b.p - a.p);
    const top = list.slice(0, SUSPECTS_SENT), shared = samePlace(reply, top.map(s => s.i));
    this.kept = { cell: this.cell, list: top.map(({ i, p }) => ({ x: +reply.x[i].toFixed(2), y: +reply.y[i].toFixed(2), gene: reply.genes[reply.gene[i]], p,
      ...shared.get(i) })) };
  }
  /** The suspects the window last placed for cell `label` (a test of a setting starts with them): at most
   *  `max` of [{x, y, gene, p, rank?, of?}] by P; [] when it holds none for that cell. */
  suspects(label = this.cell, max = SUSPECTS_SENT) {
    const kept = this.kept;
    return kept && label > 0 && kept.cell === label ? kept.list.slice(0, Math.max(0, max)).map(s => ({ ...s })) : [];
  }
  /** Off: the stage draws as without it (Stage.setSpill(null), Stage.setGlow(null), no starting outline), a row's Show is
   *  let go, the chips get their own rule back. */
  whereOff(why = 'off') {
    if (!this.where) return;
    this.where = null; this.cancel('where-ask'); this.cancel('where-working');
    this.sendGlow(null);
    this.sendStartOutline(false);
    this.call('onSpill', null);
    this.relight();   // a row the glow lay near none of is a source row again
    if (this.pin) { if (this.hot === this.pin) this.hot = 0; this.pin = 0; this.repaintRows(); this.refocus(); }
    this.paintWhere(); this.paintWhereChips();
    this.log(`dots on the image off (${why})`);
  }
  /** The toggle's state and Where they sit: the glow's line and what cannot glow, a status line (finding, waiting,
   *  failed, the dots kept from the counted frame before, none here, 3-D, too many), then per source its genes and this
   *  cell's own genes inside it. Rebuilt only when what it says changed. */
  paintWhere() {
    const w = this.where, ctx = this.ctx();
    this.whereView3d = !!ctx.view3d; this.whereEnded = RUN_ENDED.has(ctx.run?.state);
    if (this.whereButton.getAttribute('aria-pressed') !== String(!!w)) this.whereButton.setAttribute('aria-pressed', String(!!w));
    setHidden(this.whereSection, !w);
    const state = !w ? 'off' : w.error ? w.error.kind : w.reply ? (w.reply.count ? 'shown' : 'none') : 'asking';
    if (this.whereSection.dataset.state !== state) this.whereSection.dataset.state = state;
    // The cell's starting outline is drawn dashed in 2-D only, and only for a run that started from cells (sendStartOutline).
    const outline = !this.whereView3d && ctx.startSource !== 'nuclei', kept = ctx.run?.constants?.kept ?? [];
    const model = w?.reply ? whereModel(w.reply, w.req, w.colors, w.glow, this.whereEnded, kept, outline) : null, line = model?.line ?? null;
    setText(this.glowText, line?.text ?? ''); setTitle(this.glowLegend, line?.tip ?? glowTip(outline)); setHidden(this.glowLegend, !line);
    this.paintLineCounts(line?.counts ?? null);
    setText(this.glowUnlit, model?.unlit?.text ?? ''); setTitle(this.glowUnlit, model?.unlit?.tip ?? ''); setHidden(this.glowUnlit, !model?.unlit);
    if (!w) { if (this.whereList._key != null) { this.whereList._key = null; this.whereList.replaceChildren(); } this.notes(this.whereStatus, []); return; }
    const snapped = w.snapped && !w.error ? { text: deep('deepWhereSnapped', { i: number(w.snapped.iter) }),
      tip: deep('deepWhereSnappedTooltip', { i: number(w.snapped.iter) }) } : null;
    this.notes(this.whereStatus, [w.error ? (w.error.kind === 'waiting' ? T.deepWhereWaiting : T.deepWhereFailed) : null,
      !w.reply && !w.error && w.working ? T.deepWhereWorking : null, snapped,
      model?.none ? T.deepWhereNone : null, model?.truncated ?? null, ctx.view3d ? T.deepWhere3d : null].filter(Boolean)
      .map(n => (typeof n === 'string' ? { text: n } : n)));
    const key = JSON.stringify(model && [model.sources, model.other]);
    if (this.whereList._key === key) return;
    this.whereList._key = key;
    const geneNode = (g, role) => {
      const p = this.el('p', `qc-deeper-where-${role === 'own' ? 'into' : 'gene'}`); p.dataset.role = role;
      if (role === 'own') p.dataset.genes = g.genes.join(' '); else p.dataset.gene = g.gene;
      if (g.word) p.dataset.word = g.word;
      if (g.tooltip) p.title = g.tooltip;
      const dot = this.el('i', 'qc-deeper-where-dot'); dot.style.background = g.color;
      p.append(dot, this.el('span', 'qc-deeper-where-word', g.text));
      return p;
    };
    const blocks = (model?.sources ?? []).map(src => {
      const div = this.el('div', 'qc-deeper-where-source'); div.dataset.cell = src.away ? '' : src.name;
      if (src.away) div.dataset.away = '1';   // a cell with no outline here (none of the glow lies near the neighbour)
      const head = this.el('p', 'qc-deeper-where-head', src.name); head.title = src.head;
      div.append(head, ...src.genes.map(g => geneNode(g, 'from')));
      if (src.into) div.append(geneNode(src.into, 'own'));
      return div;
    });
    if (model?.other.length) {
      const div = this.el('div', 'qc-deeper-where-source'); div.dataset.cell = '';
      div.append(this.el('p', 'qc-deeper-where-head', T.deepWhereOther), ...model.other.map(g => geneNode(g, 'from')));
      blocks.push(div);
    }
    this.whereList.replaceChildren(...blocks);
  }
  /** The line's numbers on its node, for the harness (data-likely, then data-own, data-none, data-giver, data-other and
   *  data-unknown once the places are known; none while the line is not shown). */
  paintLineCounts(counts) {
    const d = this.glowLegend.dataset;
    for (const key of ['likely', ...PLACES]) {
      const v = counts?.[key];
      if (Number.isInteger(v)) { if (d[key] !== String(v)) d[key] = String(v); }
      else if (key in d) delete d[key];
    }
  }
  /** While the dots show, each chip wears its gene's dot colour (a wider rule); otherwise the chips' own magenta rule. */
  paintWhereChips() {
    const w = this.where, colors = new Map();
    if (w?.reply) w.reply.genes.forEach((g, i) => { if (w.reply.roles[i] === 'from' && w.colors[i]) colors.set(g, w.colors[i]); });
    for (const [gene, node] of this.chipNodes ?? []) {
      const color = colors.get(gene) ?? '';
      if ((node.dataset.dot ?? '') === color) continue;
      if (color) node.dataset.dot = color; else delete node.dataset.dot;
      node.style.borderLeftColor = color;
    }
  }
  /** For the harness snapshot (qc.deeper.where): on, its state, what it asked (the harness asks the engine the same),
   *  what the stage was given (lit cells, dots per gene, colours), the engine's words, where the reply's shares came
   *  from (held), the reply's total (proseg's count of the cell in the box, as its sum of w) beside the result's own
   *  numbers for the same frame that the endpoint's invariants hold against: its transcripts + outside (proseg's count
   *  of the cell there) and each asked gene's transcripts (tx: its count of the gene there); the glow ({count, likely,
   *  near, places: {own, none, giver, other, unknown} of those its line counts, the likely ones or with none every
   *  glowing one (null until the start line's reply is in), start: places.own, levels: {nucleus, cell, unknown}, the
   *  start's hold of those inside its own starting outline (null without the start layers: a whole-sample run's), row:
   *  the label of the row whose Show is on, or null, notNear: the source none of the glow lies near, or null}; null while
   *  there is no answer); what it asked of the start line (startAsked) and whether the stage holds the starting outline
   *  (startOutline); `snapped`: the exact reply shown stands for a newer frame whose own shares would be approximate
   *  (seq, held and the rest stay the shown reply's). */
  whereSnapshot() {
    const w = this.where, rep = w?.reply ?? null;
    const held = rep ? this.results.get(rep.cell)?.get(rep.seq) : null;
    const rows = held?.state === 'ready' ? new Map(held.genes.map(g => [g.gene, g.tx])) : null;
    return { on: !!w, state: this.whereSection.dataset.state, asked: w?.req?.params ?? w?.wanted?.params ?? null, seq: rep?.seq ?? null,
      sources: w?.req ? w.req.labels.map(l => this.nameOf(l)) : [], lit: w?.shown?.lit ?? [], genes: rep?.genes ?? [], roles: rep?.roles ?? [],
      colors: rep ? w.colors : [], points: rep?.count ?? 0, counts: rep ? w.counts : {}, truncated: !!rep?.truncated,
      words: Object.fromEntries((rep?.stats ?? []).filter(s => s.word).map(s => [s.gene, s.word])), total: rep?.cell_total ?? null,
      held: rep?.held ?? null, snapped: !!w?.snapped,
      expected: rows ? held.transcripts + held.outside : null, tx: rows ? Object.fromEntries(rep.genes.filter(g => rows.has(g)).map(g => [g, rows.get(g)])) : {},
      box: rep?.box ?? null, flown: !!w?.flown, startAsked: w?.startParams ?? null,
      glow: rep && w.glow ? { ...w.glow, places: w.glow.places && { ...w.glow.places }, levels: w.glow.levels && { ...w.glow.levels } } : null,
      startOutline: !!this.startOutlineKey };
  }

  // ---- the mixing check's button ------------------------------------------------------------------------------------------
  /** "Go deeper ›" in the mixing check's row for the selected cell `label` (qc-panel.js setCell, bound through main.js):
   *  its words, state and click. */
  bindButton(button, label) {
    if (!button) return;
    const open = this.isOpen && this.cell === label;
    setText(button, open ? T.deepButtonOpen : T.deepButton);
    if (button.title !== T.deepButtonTooltip) button.title = T.deepButtonTooltip;
    button.dataset.action = 'go-deeper'; button.setAttribute('aria-pressed', String(open)); button.setAttribute('aria-controls', 'qc-deeper');
    button.onclick = () => { this.log(open ? 'Go deeper closed from the mixing check' : `Go deeper from the mixing check: ${this.nameOf(label)}`); this.toggle(label, { from: button }); };
  }

  // ---- position -----------------------------------------------------------------------------------------------------------
  /** Make smaller (a strip that stays live) or bigger, where it is. */
  setSmall(on) { this.small = !!on; this.root.classList?.toggle('qc-deeper-small', this.small); this.paintHeader(); this.moveTo(this.left ?? 0, this.top ?? 0); }
  /** Where the window goes: the remembered place (fractions of the stage), else top right under Fit; kept inside the
   *  stage. `reset`: the default place. A new window at the default place moves to the bottom right if it would cover
   *  its cell. */
  place(reset = false, label = this.cell) {
    const { width: W = 800, height: H = 600 } = this.ctx().stage ?? {};
    let saved = null;
    if (!reset) try { saved = JSON.parse(this.storage?.getItem(STORAGE_KEY) || 'null'); } catch { saved = null; }
    const w = Math.min(WIDTH, Math.max(160, W - 2 * MARGIN));
    let left, top;
    if (saved && finite(saved.x) && finite(saved.y)) { left = saved.x * W; top = saved.y * H; }
    else {
      left = W - w - RIGHT; top = TOP;
      const c = this.o.screenOf?.(label);
      const h = Math.min(this.size.h, H - top - MARGIN);
      if (c && c[0] >= left && c[0] <= left + w && c[1] >= top && c[1] <= top + h) top = Math.max(MARGIN, H - h - MARGIN);
    }
    this.moveTo(left, top);
  }
  moveTo(left, top) {
    const { width: W = 800, height: H = 600 } = this.ctx().stage ?? {};
    const w = Math.min(this.size.w || WIDTH, W - 2 * MARGIN), h = this.size.h || 200;
    left = clamp(left, MARGIN, Math.max(MARGIN, W - w - MARGIN));
    top = clamp(top, MARGIN, Math.max(MARGIN, H - Math.min(h, 120) - MARGIN));
    this.left = Math.round(left); this.top = Math.round(top);
    const s = this.root.style;
    if (s.left !== `${this.left}px`) s.left = `${this.left}px`;
    if (s.top !== `${this.top}px`) s.top = `${this.top}px`;
    const max = `${Math.max(120, Math.round(H - this.top - MARGIN))}px`;
    if (s.maxHeight !== max) s.maxHeight = max;
    this.box();
  }
  savePlace() {
    const { width: W = 0, height: H = 0 } = this.ctx().stage ?? {};
    if (!(W > 0 && H > 0)) return;
    try { this.storage?.setItem(STORAGE_KEY, JSON.stringify({ x: +(this.left / W).toFixed(4), y: +(this.top / H).toFixed(4) })); } catch { /* optional storage */ }
  }
  forgetPlace() { try { this.storage?.removeItem?.(STORAGE_KEY); } catch { /* optional storage */ } }
  /** The stage changed size: the window stays inside it (at most a comparison per call: drawOverlay calls it often). */
  layout() {
    if (!this.isOpen) return;
    const ctx = this.ctx(), { width: W, height: H } = ctx.stage ?? {};
    if (this.where && !!ctx.view3d !== this.whereView3d) this.paintWhere();   // the genes' dots show in 2-D only: said in 3-D
    if (W === this.stageW && H === this.stageH) return;
    this.stageW = W; this.stageH = H;
    this.moveTo(this.left ?? 0, this.top ?? 0);
  }
  box() {
    if (!this.isOpen) return;
    const b = { left: this.left ?? 0, top: this.top ?? 0, w: this.size.w || WIDTH, h: this.size.h || 0 };
    const key = `${b.left},${b.top},${Math.round(b.w)},${Math.round(b.h)}`;
    if (key === this.boxKey) return;
    this.boxKey = key; this.call('onBox', b);
  }

  // ---- the harness ------------------------------------------------------------------------------------------------------
  /** For the harness snapshot (qc.deeper): what the window shows now. */
  snapshot() {
    const m = this.shownModel?.state === 'ready' ? this.shownModel : null, r = this.shown?.result ?? null;
    return {
      open: this.isOpen, cell: this.isOpen ? this.nameOf(this.cell) : null, label: this.isOpen ? this.cell : 0,
      seq: r?.seq ?? null, iter: r?.iter ?? null, state: this.isOpen ? this.root.dataset.state : 'closed',
      code: m ? m.code : null, shown: m ? m.p : null, parts: m && r ? r.parts.map(v => +v.toFixed(4)) : null,
      bar: m ? [m.bar.own, m.bar.nb, m.bar.looseUnder ? '<1' : m.bar.loose] : null,
      learned: m ? m.learned : null, foreign: m && !m.none ? this.order.chips.slice(0, this.chipsAll ? undefined : CHIPS) : [],
      sources: m && !m.none ? this.order.rows.map(k => { const n = m.neighbours.find(x => x.key === k); return n ? [n.name, +(n.share * 100).toFixed(1)] : null; }).filter(Boolean) : [],
      small: this.small, rough: !!r?.rough, tag: this.shown?.tag ?? '', transcripts: r?.transcripts ?? null, asked: this.asked,
      frame: this.ctx().seq ?? null, focus: this.focus ? { cell: this.focus.cell, sources: this.focus.sources.map(s => s.label), same: this.focus.same,
        lit: this.focus.lit ?? [], dim: this.focus.dim ?? 0 } : null,
      place: this.isOpen ? [this.left, this.top] : null, where: this.whereSnapshot(),
      // "Test a setting here…" offered beside Show on the image, or beside the check's switch while it is off (the page
      // can test this cell); never while made smaller (its CSS hides the entry).
      test: this.isOpen && !this.small && !this.testButton.hidden,
      // The basis's tooltip as shown (its background, once the frame's statistics are in).
      basis: this.isOpen && m ? { text: this.basisNode.textContent, tip: this.basisNode.title } : null,
    };
  }
  /** Every word the window shows now (text, tooltips and labels), for the harness's banned-word check. */
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
  /** Every string the window can show (tests). */
  static allTexts() { return allTexts(); }
  dispose() {
    this.reset(null);
    for (const [node, type, fn, opts] of this.listeners) node?.removeEventListener?.(type, fn, opts);
    this.listeners = []; this.observer?.disconnect?.(); this.root.remove?.();
  }
}

/** Each result held, its own number (update's key): one that replaces another for the same frame and cell (another
 *  model, or another answer) is a new update, painted as one, its glow with it. */
const SERIALS = new WeakMap();
let serials = 0;
function serialOf(r) { let n = SERIALS.get(r); if (n === undefined) SERIALS.set(r, n = ++serials); return n; }
/** What tells two answers for the same frame and cell apart (one of another count too: a frame counted again for the
 *  hold, a finish or the end is answered on the count proseg saves, and that answer replaces the one held). */
function signature(r) {
  return [r.state, r.model, r.count, r.iter, r.code, r.transcripts, r.genes?.length, r.genes?.[0]?.nb ?? r.genes?.[0]?.[1], r.neighbours?.length,
    r.neighbours?.[0]?.tx, r.same_kind_share, r.learned].join('|');
}
function setText(node, text) { text = text == null ? '' : String(text); if (node && node.textContent !== text) node.textContent = text; }
function setHidden(node, hidden) { if (node && node.hidden !== hidden) node.hidden = hidden; }
function setTitle(node, text) { text = text == null ? '' : String(text); if (node && node.title !== text) node.title = text; }
